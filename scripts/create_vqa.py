from __future__ import annotations

import json
import logging
import os
import random
import uuid
from argparse import ArgumentParser
from typing import Any, Literal

import pandas as pd
from tqdm import tqdm

from utils.preprocessing import NOT_PRESENT, flatten_dict
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
    FIELD_SPECS,
    get_negative_candidates,
    sample_negative_candidates_one_per_field,
)


class ContextBuilder:
    def __init__(self, seed: int = 42, shuffle: bool = False, minimal_prob: float = 0.5):
        self.rng = random.Random(seed)
        self.shuffle = shuffle
        self.minimal_prob = minimal_prob

    def get_and_append(self, sentences: list[str], field_str: str, dict_map: dict[str, Any], templates: list[str]):
        field_value = dict_map.get(field_str, None)
        if field_value is not None:
            t = self.rng.choice(templates)
            sentences.append(t.format(value=field_value))

    def build_context(self, context: dict) -> str:
        patient = context.get("patient") or {}
        exam = context.get("exam") or {}
        sentences: list[str] = []

        self.get_and_append(sentences, "age", patient, AGE_TEMPLATES)
        implants = patient.get("has_implants", None)
        if implants is not None:
            tmpl = IMPLANTS_YES_TEMPLATES if implants == "yes" else IMPLANTS_NO_TEMPLATES
            sentences.append(self.rng.choice(tmpl))

        self.get_and_append(sentences, "modality", exam, MODALITY_TEMPLATES)
        self.get_and_append(sentences, "view", exam, VIEW_TEMPLATES)
        self.get_and_append(sentences, "contrast_phase", exam, PHASE_TEMPLATES)
        laterality = exam.get("laterality", None)
        if laterality is not None:
            if laterality == "bilateral":
                sentences.append(self.rng.choice(BILATERAL_TEMPLATES))
            else:
                sentences.append(self.rng.choice(LATERALITY_TEMPLATES).format(value=laterality))

        if self.minimal_prob > 0:
            sentences = [s for s in sentences if self.rng.random() > self.minimal_prob]

        if self.shuffle and len(sentences) > 1:
            self.rng.shuffle(sentences)

        out = " ".join(sentences).strip()

        if len(out) == 0:
            return self.rng.choice(NO_CONTEXT_TEMPLATES)

        return out


class VQAGenerator:
    def __init__(
        self,
        random_seed: int = 42,
        shuffle_context: bool = False,
        shuffle_questions: bool = False,
        minimal_context_ratio: float = 0.5,
        add_negative_examples_prob: float = 0.0,
        negative_examples_k: int = 0,
    ) -> None:
        self.rng = random.Random(random_seed)
        self.shuffle_questions = shuffle_questions
        self.add_negative_examples_prob = add_negative_examples_prob
        self.negative_examples_k = negative_examples_k
        self.context_builder = ContextBuilder(seed=random_seed, shuffle=shuffle_context, minimal_prob=minimal_context_ratio)
        self.field_specs = FIELD_SPECS
        self.field_specs_by_id = {f.field_id: f for f in self.field_specs}

    def generate_for_row(self, row: pd.Series, split: Literal["train", "val", "test"]) -> dict[str, Any]:
        context = json.loads(row["context"])
        context_str = self.context_builder.build_context(context)

        findings = json.loads(row["findings"])
        flatten_findings = flatten_dict(findings)

        conversation = []
        for field, value in flatten_findings.items():
            if value == NOT_PRESENT:
                continue

            fs = self.field_specs_by_id[field]
            question = self.rng.choice(fs.templates)
            answer = str(value)
            conversation.append({"role": "user", "text": question})
            conversation.append(
                {
                    "role": "assistant",
                    "text": answer,
                    "task_type": fs.task_type,
                    "field_id": fs.field_id,
                    "scope": fs.scope,
                    "difficulty": fs.difficulty,
                    "answer_source": fs.answer_source,
                    "is_negative": False,
                }
            )

        # Add a question regarding the presence of lesions
        if "lesion" in findings:
            fs = self.field_specs_by_id["lesion.presence"]
            question = self.rng.choice(fs.templates)
            conversation.append({"role": "user", "text": question})
            conversation.append(
                {
                    "role": "assistant",
                    "text": "no" if findings["lesion"] == NOT_PRESENT else "yes",
                    "task_type": fs.task_type,
                    "field_id": fs.field_id,
                    "scope": fs.scope,
                    "difficulty": fs.difficulty,
                    "answer_source": fs.answer_source,
                    "is_negative": False,
                }
            )

        # Add negative examples if specified
        if self.add_negative_examples_prob > 0 and self.negative_examples_k > 0 and self.rng.random() < self.add_negative_examples_prob:
            lp = 0 if "lesion" not in findings else 1 if findings["lesion"] != NOT_PRESENT else -1
            candidates = get_negative_candidates(
                field_specs_by_id=self.field_specs_by_id,
                modality=row["modality"],
                lesions_present=lp,
                lesion_type=findings["lesion"]["type"] if "lesion" in findings and "type" in findings["lesion"] else "",
            )

            sampled_candidates = sample_negative_candidates_one_per_field(
                rng=self.rng,
                candidates=candidates,
                k=self.negative_examples_k,
            )

            for cand in sampled_candidates:
                fs = self.field_specs_by_id[cand.field_id]
                question = self.rng.choice(fs.templates)

                conversation.append({"role": "user", "text": question})
                conversation.append(
                    {
                        "role": "assistant",
                        "text": cand.answer,
                        "task_type": fs.task_type,
                        "field_id": fs.field_id,
                        "scope": fs.scope,
                        "difficulty": fs.difficulty,
                        "answer_source": fs.answer_source,
                        "is_negative": True,
                        "negative_case": cand.case,
                    }
                )

        if self.shuffle_questions:
            # Shuffle only user turns and their corresponding assistant turns together
            pairs = [(conversation[i], conversation[i + 1]) for i in range(0, len(conversation), 2)]
            self.rng.shuffle(pairs)
            conversation = [item for pair in pairs for item in pair]

        assistant_turns = [t for t in conversation if t["role"] == "assistant"]
        return {
            "id": f"{row['id']}_dialogue_{uuid.uuid4()}",
            "source_id": str(row["id"]),
            "split": split,
            "image_path": str(row["exam"]),
            "context": context_str,
            "conversation": conversation,
            "metadata": {
                "question_count": len(assistant_turns),
                "has_negative": any(t.get("is_negative", False) for t in assistant_turns),
                "negative_count": sum(int(t.get("is_negative", False)) for t in assistant_turns),
                "lesion_status": (
                    "unknown" if "lesion" not in flatten_findings else "present" if flatten_findings["lesion"] != NOT_PRESENT else "absent"
                ),
                "answer_sources": sorted({t.get("answer_source", "image") for t in assistant_turns}),
            },
        }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="../data/report_generation_split_vqa")
    parser.add_argument("--output_fn", default="vqa.jsonl")
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--shuffle_context", action="store_true")
    parser.add_argument("--shuffle_questions", action="store_true")
    parser.add_argument("--minimal_context_ratio", type=float, default=0.5)
    parser.add_argument("--add_negative_examples_prob", type=float, default=0.0)
    parser.add_argument("--negative_examples_k", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/convert_to_vqa.log",
        filemode="a",
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    generator = VQAGenerator(
        random_seed=args.random_seed,
        shuffle_context=args.shuffle_context,
        shuffle_questions=args.shuffle_questions,
        minimal_context_ratio=args.minimal_context_ratio,
        add_negative_examples_prob=args.add_negative_examples_prob,
        negative_examples_k=args.negative_examples_k,
    )
    examples = [generator.generate_for_row(row, args.split) for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating VQA examples")]

    out_path = os.path.join(args.output_dir, args.output_fn)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in examples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logging.info("Done. Total VQA examples: %d", len(examples))
