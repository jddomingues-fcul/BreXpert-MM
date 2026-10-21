from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from argparse import ArgumentParser
from collections import Counter
from typing import Any, Literal

import pandas as pd
from tqdm import tqdm

from dtos.json_report_findings import (
    LESION_STATUS_ABSENT,
    LESION_STATUS_PRESENT,
    LESION_STATUS_UNKNOWN,
    SCOPE_COMPLETE,
)
from utils.vqa_context_templates import (
    AGE_TEMPLATES,
    BILATERAL_TEMPLATES,
    IMPLANTS_NO_TEMPLATES,
    IMPLANTS_YES_TEMPLATES,
    LATERALITY_TEMPLATES,
    MODALITY_TEMPLATES,
    NO_CONTEXT_TEMPLATES,
    PHASE_TEMPLATES,
    VIEW_TEMPLATES,
)
from utils.vqa_findings_templates import (
    FIELD_SPECS_BY_ID,
    STAGE_ASSESSMENT,
    STAGE_COUNT,
    STAGE_PRESENCE,
    get_negative_candidates,
    sample_negative_candidates_one_per_field,
)

logger = logging.getLogger(__name__)

# Which findings-record key holds each per-finding detail block. These are the
# only payloads `Lesion` defines, so they are the only blocks a record can carry.
DETAIL_BLOCKS = (
    ("mass details", "lesion.mass details."),
    ("calcification details", "lesion.calcification details."),
)

GATING_CONTEXT_KEYS = frozenset(
    key for spec in FIELD_SPECS_BY_ID.values() for key in spec.requires_context
)

# Attribute families. A per-finding question never names the family it asks
# about, so a referent cannot contain its own answer: asking the finding type
# identifies by location, asking a location identifies by type and descriptors,
# asking a descriptor identifies by type and location.
FAMILY_TYPE = "type"
FAMILY_LOCATION = "location"
FAMILY_DESCRIPTOR = "descriptor"
FAMILIES = (FAMILY_TYPE, FAMILY_LOCATION, FAMILY_DESCRIPTOR)

# A finding that is the only one of its kind in the image needs no referent at
# all -- a radiologist looking at a single mass asks "what is the shape?", not
# "what is the shape of the mass at 10 o'clock of the right breast?".
NO_REFERENT = ""

# Per family, per finding, the candidate referents from shortest to most
# specific, each paired with the attributes it asserts.
Referents = dict[str, list[list[tuple[str, dict[str, str]]]]]


def family_for(field_id: str) -> str:
    if field_id == "lesion.finding type":
        return FAMILY_TYPE
    if field_id.startswith("lesion.location."):
        return FAMILY_LOCATION
    return FAMILY_DESCRIPTOR


class ContextBuilder:
    def __init__(self, shuffle: bool = False, dropout: float = 0.0):
        self.shuffle = shuffle
        self.dropout = dropout

    @staticmethod
    def _identifiers(block: Any) -> dict:
        if not isinstance(block, dict):
            return {}
        return {str(key).replace(" ", "_"): value for key, value in block.items()}

    def build(self, context: Any, rng: random.Random) -> tuple[str, set[str]]:
        context = context if isinstance(context, dict) else {}
        patient = self._identifiers(context.get("patient"))
        exam = self._identifiers(context.get("exam"))
        entries: list[tuple[str, str]] = []

        age = patient.get("age")
        if age is not None:
            entries.append(("age", rng.choice(AGE_TEMPLATES).format(value=age)))

        implants = patient.get("has_implants")
        if implants is not None:
            templates = (
                IMPLANTS_YES_TEMPLATES if implants == "yes" else IMPLANTS_NO_TEMPLATES
            )
            entries.append(("has_implants", rng.choice(templates)))

        modality = exam.get("modality")
        if modality is not None:
            entries.append(
                ("modality", rng.choice(MODALITY_TEMPLATES).format(value=modality))
            )

        view = exam.get("view")
        if view is not None:
            entries.append(("view", rng.choice(VIEW_TEMPLATES).format(value=view)))

        phase = exam.get("contrast_phase")
        if phase is not None:
            entries.append(
                ("contrast_phase", rng.choice(PHASE_TEMPLATES).format(value=phase))
            )

        laterality = exam.get("laterality")
        if laterality is not None:
            sentence = (
                rng.choice(BILATERAL_TEMPLATES)
                if laterality == "bilateral"
                else rng.choice(LATERALITY_TEMPLATES).format(value=laterality)
            )
            entries.append(("laterality", sentence))

        if self.dropout > 0:
            entries = [
                entry
                for entry in entries
                if rng.random() >= self.dropout or entry[0] in GATING_CONTEXT_KEYS
            ]

        if self.shuffle and len(entries) > 1:
            rng.shuffle(entries)

        available = {key for key, _ in entries}
        text = " ".join(sentence for _, sentence in entries).strip()
        if not text:
            return rng.choice(NO_CONTEXT_TEMPLATES), available
        return text, available


class VQAGenerator:
    def __init__(
        self,
        random_seed: int = 42,
        shuffle_context: bool = False,
        shuffle_questions: bool = False,
        context_dropout: float = 0.0,
        add_negative_examples_prob: float = 0.0,
        negative_examples_k: int = 0,
        max_turns: int = 0,
        multiple_choice_prob: float = 0.0,
    ) -> None:
        self.random_seed = random_seed
        self.shuffle_questions = shuffle_questions
        self.add_negative_examples_prob = add_negative_examples_prob
        self.negative_examples_k = negative_examples_k
        self.max_turns = max_turns
        self.multiple_choice_prob = multiple_choice_prob
        self.context_builder = ContextBuilder(
            shuffle=shuffle_context, dropout=context_dropout
        )
        self.field_specs_by_id = FIELD_SPECS_BY_ID
        self.stats: Counter = Counter()

    # -- determinism ---------------------------------------------------------
    def _rng_for(self, row_id: str) -> random.Random:
        digest = hashlib.sha256(f"{self.random_seed}:{row_id}".encode()).hexdigest()
        return random.Random(int(digest[:16], 16))

    # -- turn construction ---------------------------------------------------
    def _ask(
        self,
        rng: random.Random,
        spec,
        answer: str,
        modality: str,
        referent: str | None = None,
        lesion_id: str | None = None,
        is_negative: bool = False,
        negative_case: str | None = None,
    ) -> tuple[dict, dict]:
        templates = spec.templates
        if referent == NO_REFERENT and spec.solo_templates:
            templates = spec.solo_templates
        question = rng.choice(templates).format(referent=referent or "the finding")

        options = spec.answers_for(modality)
        rendered_options = None
        if (
            options
            and not is_negative
            and self.multiple_choice_prob > 0
            and rng.random() < self.multiple_choice_prob
        ):
            shuffled = list(options)
            rng.shuffle(shuffled)
            rendered_options = shuffled
            question = f"{question} Options: {'; '.join(shuffled)}."

        user_turn = {"role": "user", "text": question}
        assistant_turn = {
            "role": "assistant",
            "text": answer,
            "task_type": spec.task_type,
            "field_id": spec.field_id,
            "scope": spec.scope,
            "difficulty": spec.difficulty,
            "answer_source": spec.answer_source,
            "stage": spec.stage,
            "is_negative": is_negative,
            "lesion_id": lesion_id,
            "answer_space": list(options) if options else None,
            "options_shown": rendered_options,
        }
        if negative_case is not None:
            assistant_turn["negative_case"] = negative_case
        return user_turn, assistant_turn

    def _sample_turns(
        self, rng, findings: dict, modality: str, available_context: set[str]
    ) -> list[tuple[dict, dict]]:
        turns = []
        breast = findings.get("breast") or {}
        lesion_status = findings.get("lesion status", LESION_STATUS_UNKNOWN)

        for key, field_id in (
            ("composition", "breast.composition"),
            ("fgt", "breast.fgt"),
            ("tissue composition", "breast.tissue composition"),
        ):
            value = breast.get(key)
            if value is None:
                continue
            spec = self.field_specs_by_id[field_id]
            if not self._context_satisfied(spec, available_context):
                continue
            turns.append(self._ask(rng, spec, str(value), modality))

        if lesion_status in (LESION_STATUS_PRESENT, LESION_STATUS_ABSENT):
            spec = self.field_specs_by_id["lesion.presence"]
            turns.append(
                self._ask(
                    rng,
                    spec,
                    "yes" if lesion_status == LESION_STATUS_PRESENT else "no",
                    modality,
                )
            )

        # Counting is only answerable when the source annotated everything.
        lesions = findings.get("lesions") or []
        scope = findings.get("annotation scope")
        if lesion_status == LESION_STATUS_ABSENT:
            spec = self.field_specs_by_id["lesion.count"]
            turns.append(self._ask(rng, spec, "0", modality))
        elif (
            lesion_status == LESION_STATUS_PRESENT
            and scope == SCOPE_COMPLETE
            and lesions
        ):
            spec = self.field_specs_by_id["lesion.count"]
            turns.append(self._ask(rng, spec, str(len(lesions)), modality))

        assessment = findings.get("assessment") or {}
        if assessment.get("birads") is not None:
            spec = self.field_specs_by_id["assessment.birads"]
            if self._context_satisfied(spec, available_context):
                turns.append(self._ask(rng, spec, str(assessment["birads"]), modality))

        return turns

    @staticmethod
    def _context_satisfied(spec, available_context: set[str]) -> bool:
        return all(key in available_context for key in spec.requires_context)

    @staticmethod
    def _also_fits(described: dict[str, str], other: dict[str, str]) -> bool:
        return all(other.get(key, value) == value for key, value in described.items())

    def _referent_or_none(
        self, referents: Referents, index: int, field_id: str
    ) -> str | None:
        """The shortest referent no other finding in the image also matches."""
        ladders = referents[family_for(field_id)]
        others = [
            ladder[-1][1] for other, ladder in enumerate(ladders) if other != index
        ]
        for phrase, attrs in ladders[index]:
            if not any(self._also_fits(attrs, other) for other in others):
                return phrase
        self.stats["turns_skipped_ambiguous_referent"] += 1
        return None

    def _lesion_turns(
        self,
        rng,
        lesion: dict,
        referents: Referents,
        index: int,
        modality: str,
        available: set[str],
    ) -> list[tuple[dict, dict]]:
        turns = []
        lesion_id = lesion.get("lesion id")

        finding_type = lesion.get("finding type")
        if finding_type is not None:
            field_id = "lesion.finding type"
            referent = self._referent_or_none(referents, index, field_id)
            if referent is not None:
                spec = self.field_specs_by_id[field_id]
                turns.append(
                    self._ask(
                        rng, spec, str(finding_type), modality, referent, lesion_id
                    )
                )

        location = lesion.get("location") or {}
        for key, field_id in (
            ("laterality", "lesion.location.laterality"),
            ("quadrant", "lesion.location.quadrant"),
            ("clock position", "lesion.location.clock position"),
            ("depth", "lesion.location.depth"),
            ("special location", "lesion.location.special location"),
        ):
            value = location.get(key)
            if value is None:
                continue
            spec = self.field_specs_by_id[field_id]
            if not self._context_satisfied(spec, available):
                self.stats["turns_skipped_missing_context"] += 1
                continue
            referent = self._referent_or_none(referents, index, field_id)
            if referent is None:
                continue
            turns.append(
                self._ask(rng, spec, str(value), modality, referent, lesion_id)
            )

        for block_key, field_prefix in DETAIL_BLOCKS:
            block = lesion.get(block_key) or {}
            for key, value in block.items():
                if value is None or value is False:
                    continue
                field_id = f"{field_prefix}{key}"
                spec = self.field_specs_by_id.get(field_id)
                if spec is None:
                    logger.warning(f"No FieldSpec for '{field_id}'; skipping it.")
                    self.stats["turns_skipped_no_spec"] += 1
                    continue
                if not self._context_satisfied(spec, available):
                    self.stats["turns_skipped_missing_context"] += 1
                    continue
                referent = self._referent_or_none(referents, index, field_id)
                if referent is None:
                    continue
                turns.append(
                    self._ask(rng, spec, str(value), modality, referent, lesion_id)
                )

        return turns

    def _negative_turns(self, rng, findings, modality, available):
        if not (self.add_negative_examples_prob > 0 and self.negative_examples_k > 0):
            return []
        if rng.random() >= self.add_negative_examples_prob:
            return []

        lesions = findings.get("lesions") or []
        candidates = get_negative_candidates(
            field_specs_by_id=self.field_specs_by_id,
            modality=modality,
            lesion_status=findings.get("lesion status", LESION_STATUS_UNKNOWN),
            lesion_types=tuple(lesion.get("finding type") for lesion in lesions),
            annotation_scope=findings.get("annotation scope", "unknown"),
        )
        sampled = sample_negative_candidates_one_per_field(
            rng=rng, candidates=candidates, k=self.negative_examples_k
        )

        turns = []
        for candidate in sampled:
            spec = self.field_specs_by_id[candidate.field_id]
            if not self._context_satisfied(spec, available):
                continue
            turns.append(
                self._ask(
                    rng,
                    spec,
                    candidate.answer,
                    modality,
                    referent="the finding",
                    is_negative=True,
                    negative_case=candidate.case,
                )
            )
        return turns

    # -- entry point ---------------------------------------------------------
    def generate_for_row(
        self, row: pd.Series, split: Literal["train", "val", "test"]
    ) -> dict[str, Any]:
        rng = self._rng_for(str(row["id"]))

        context = json.loads(row["context"])
        context_str, available_context = self.context_builder.build(context, rng)

        findings = json.loads(row["findings"])
        modality = str(row["modality"])

        pairs = self._sample_turns(rng, findings, modality, available_context)

        lesions = findings.get("lesions") or []
        referents = {
            family: [self._referent_ladder(lesion, family) for lesion in lesions]
            for family in FAMILIES
        }
        for index, lesion in enumerate(lesions):
            pairs.extend(
                self._lesion_turns(
                    rng, lesion, referents, index, modality, available_context
                )
            )

        pairs.extend(self._negative_turns(rng, findings, modality, available_context))

        pairs = self._order(rng, pairs)

        if self.max_turns and len(pairs) > self.max_turns:
            keep_stages = {STAGE_PRESENCE, STAGE_COUNT, STAGE_ASSESSMENT}
            spine = [p for p in pairs if p[1]["stage"] in keep_stages]
            spine = spine[: self.max_turns]
            rest = [p for p in pairs if p[1]["stage"] not in keep_stages]
            budget = max(0, self.max_turns - len(spine))
            rest = rng.sample(rest, k=min(budget, len(rest)))
            pairs = self._order(rng, spine + rest)
            self.stats["dialogues_truncated"] += 1

        conversation = [turn for pair in pairs for turn in pair]
        assistant_turns = [pair[1] for pair in pairs]
        self.stats["turns"] += len(assistant_turns)
        self.stats["dialogues"] += 1
        dialogue_hash = hashlib.sha256(
            json.dumps([context_str, conversation], sort_keys=True).encode()
        ).hexdigest()[:16]

        return {
            "id": f"{row['id']}_dialogue_{split}_{dialogue_hash}",
            "source_id": str(row["id"]),
            "split": split,
            "image_path": str(row["exam"]),
            "context": context_str,
            "conversation": conversation,
            "metadata": {
                "question_count": len(assistant_turns),
                "has_negative": any(t["is_negative"] for t in assistant_turns),
                "negative_count": sum(int(t["is_negative"]) for t in assistant_turns),
                "lesion_status": findings.get("lesion status", LESION_STATUS_UNKNOWN),
                "annotation_scope": findings.get("annotation scope"),
                "lesion_count": len(lesions),
                "assessment_scope": row.get("assessment_scope"),
                "context_available": sorted(available_context),
                "answer_sources": sorted({t["answer_source"] for t in assistant_turns}),
                "stages": sorted({t["stage"] for t in assistant_turns}),
            },
        }

    @staticmethod
    def _referent_ladder(
        lesion: dict, exclude: str
    ) -> list[tuple[str, dict[str, str]]]:
        """Ways to refer to one finding, shortest first.

        Each rung carries the attributes it asserts, so the caller can stop at
        the first one that no other finding in the image answers to. Naming more
        than that is how a report reads, not how a radiologist speaks.
        """
        type_attrs: dict[str, str] = {}
        noun = "finding"
        finding_type = lesion.get("finding type")
        if exclude != FAMILY_TYPE and finding_type:
            type_attrs["finding type"] = str(finding_type)
            noun = (
                "group of calcifications"
                if finding_type == "calcification"
                else str(finding_type)
            )

        where = None
        if exclude != FAMILY_LOCATION:
            location = lesion.get("location") or {}
            if location.get("clock position") is not None:
                where = f"{location['clock position']} o'clock"
            elif location.get("quadrant"):
                where = f"the {location['quadrant']} quadrant"
            elif location.get("special location"):
                where = f"the {location['special location']} region"
            if where and location.get("laterality") in ("left", "right"):
                where = f"{where} of the {location['laterality']} breast"

        if exclude == FAMILY_TYPE:
            # "abnormality", not "finding": the type templates already say
            # "finding", and asking what type of finding "the finding at 10
            # o'clock" is reads as a stutter.
            ladder = [(NO_REFERENT, {})]
            if where:
                ladder.append((f"the abnormality at {where}", {"where": where}))
            return ladder

        ladder = [(f"the {noun}", type_attrs)]
        if exclude == FAMILY_LOCATION:
            mass = lesion.get("mass details") or {}
            for attribute in ("margin", "shape"):
                if mass.get(attribute):
                    ladder.append(
                        (
                            f"the {mass[attribute]} {noun}",
                            {
                                **type_attrs,
                                f"mass details.{attribute}": str(mass[attribute]),
                            },
                        )
                    )
                    break
        elif where:
            ladder.append((f"the {noun} at {where}", {**type_attrs, "where": where}))
        return ladder

    def _order(self, rng, pairs):
        ordered = list(pairs)
        if self.shuffle_questions:
            rng.shuffle(ordered)
        ordered.sort(key=lambda pair: pair[1]["stage"])
        return ordered


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="../data/report_generation_split_vqa")
    parser.add_argument("--output_fn", default="vqa.jsonl")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--shuffle_context", action="store_true")
    parser.add_argument("--shuffle_questions", action="store_true")
    parser.add_argument(
        "--context_dropout",
        type=float,
        default=0.0,
        help=(
            "Probability of withholding each context sentence. Sentences that "
            "some question needs to be answerable at all (see "
            "FieldSpec.requires_context) are exempt, so dropout varies what the "
            "model is told without deleting questions. Held at 0 for val/test."
        ),
    )
    parser.add_argument("--add_negative_examples_prob", type=float, default=0.0)
    parser.add_argument("--negative_examples_k", type=int, default=0)
    parser.add_argument(
        "--multiple_choice_prob",
        type=float,
        default=0.0,
        help="Probability of rendering a closed-vocabulary question with its options.",
    )
    parser.add_argument(
        "--max_turns",
        type=int,
        default=0,
        help=(
            "Cap on assistant turns per dialogue (0 = uncapped). A cap keeps the "
            "text budget predictable when training on a single 24 GB card."
        ),
    )
    parser.add_argument(
        "--eval_preset",
        action="store_true",
        help=(
            "val/test preset: full context, no question shuffling, no negatives, "
            "so the evaluation set is deterministic and every question is "
            "answerable from what the model was given."
        ),
    )
    parser.add_argument("--debug", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/convert_to_vqa.log",
        filemode="a",
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    os.makedirs(args.output_dir, exist_ok=True)

    if args.eval_preset:
        args.context_dropout = 0.0
        args.shuffle_questions = False
        args.add_negative_examples_prob = 0.0
        args.negative_examples_k = 0

    df = pd.read_csv(args.input_csv)
    generator = VQAGenerator(
        random_seed=args.random_seed,
        shuffle_context=args.shuffle_context,
        shuffle_questions=args.shuffle_questions,
        context_dropout=args.context_dropout,
        add_negative_examples_prob=args.add_negative_examples_prob,
        negative_examples_k=args.negative_examples_k,
        max_turns=args.max_turns,
        multiple_choice_prob=args.multiple_choice_prob,
    )
    examples = [
        generator.generate_for_row(row, args.split)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating VQA examples")
    ]

    out_path = os.path.join(args.output_dir, args.output_fn)
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(item, ensure_ascii=False) + "\n" for item in examples)

    summary = (
        f"Done. {len(examples):,} dialogues, {generator.stats['turns']:,} turns "
        f"(mean {generator.stats['turns'] / max(1, len(examples)):.1f}/dialogue). "
        f"Skipped for missing context: "
        f"{generator.stats['turns_skipped_missing_context']:,}. "
        f"Skipped for ambiguous referent: "
        f"{generator.stats['turns_skipped_ambiguous_referent']:,}. "
        f"Truncated dialogues: {generator.stats['dialogues_truncated']:,}."
    )
    logging.info(summary)
    print(summary)
