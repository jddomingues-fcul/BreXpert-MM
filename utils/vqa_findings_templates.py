"""
1. Every field declares its ANSWER SPACE.

2. Every field declares its position in a CLINICAL DRILL-DOWN and what it
   depends on. The dialogue walks composition -> presence -> count -> per-finding
   type -> location -> descriptors -> assessment, so a wrong answer is
   attributable to a step and "accuracy given the parent turn was right" is
   computable.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from utils.preprocessing import (
    ASYMMETRY,
    BREAST_COMPOSITION_ORDER,
    CALCIFICATION,
    CALCIFICATION_DISTRIBUTION_ORDER,
    DEPTHS,
    ECHOGENIC_FEATURES,
    MAMMOGRAPHY,
    MARGIN_ORDER,
    MASS,
    MASS_DENSITY_ORDER,
    MASS_ECHO_PATTERN_ORDER,
    MASS_SHAPE_ORDER,
    MRI,
    MRI_FGT_ORDER,
    NON_MASS_ENHANCEMENT,
    NON_MASS_LESION,
    POSTERIOR_FEATURES_ORDER,
    QUADRANTS,
    SPECIAL_LOCATIONS,
    SUSPICIOUS_CALCIFICATIONS,
    TISSUE_COMPOSITION_VALUES,
    TYPICALLY_BENIGN_CALCIFICATIONS,
    ULTRASOUND,
)

AnswerSource = Literal["image", "image+context"]
NegativeCase = Literal["no_lesion", "wrong_modality", "wrong_lesion_type"]
Scope = Literal["sample", "lesion"]

# Which context sentence a question needs in order to be answerable. The context
# builder drops sentences at random to vary how much is given; a question whose
# supporting sentence was dropped is unanswerable and must not be asked. These
# keys match the fields the context builder reads.
CTX_LATERALITY = "laterality"

# Stages of the drill-down, asked in this order.
STAGE_COMPOSITION = 1
STAGE_PRESENCE = 2
STAGE_COUNT = 3
STAGE_FINDING_TYPE = 4
STAGE_LOCATION = 5
STAGE_DESCRIPTOR = 6
STAGE_ASSESSMENT = 7


BIRADS_ANSWER_SPACE = (
    "(1) negative",
    "(2) benign",
    "(3) probably benign",
    "(4) suspicious",
    "(5) highly suggestive of malignancy",
)

CALCIFICATION_TYPE_FAMILIES = ("typically benign", "suspicious morphology")
CALCIFICATION_MORPHOLOGIES = TYPICALLY_BENIGN_CALCIFICATIONS + SUSPICIOUS_CALCIFICATIONS

# Mass margin differs by modality in v2025, so its answer space is a mapping
# rather than a flat tuple.
MARGIN_ANSWER_SPACE = dict(MARGIN_ORDER)


@dataclass(frozen=True)
class FieldSpec:
    field_id: str
    task_type: str
    templates: list[str]

    solo_templates: list[str] | None = None

    answer_space: tuple[str, ...] | dict[str, tuple[str, ...]] | None = None

    difficulty: str = "medium"
    scope: Scope = "sample"
    answer_source: AnswerSource = "image"
    stage: int = STAGE_DESCRIPTOR

    # Context keys without which the question cannot be answered.
    requires_context: tuple[str, ...] = ()

    negative_response_templates: dict[NegativeCase, str] | None = None

    def answers_for(self, modality: str) -> tuple[str, ...] | None:
        if self.answer_space is None:
            return None
        if isinstance(self.answer_space, dict):
            return self.answer_space.get(modality)
        return self.answer_space


FIELD_SPECS: list[FieldSpec] = [
    # ---------------------------------------------------------------- breast
    FieldSpec(
        field_id="breast.composition",
        task_type="breast_composition",
        templates=[
            "What is the breast composition?",
            "Which breast composition category applies?",
            "How is the breast composition classified?",
            "Breast composition?",
            "What ACR breast composition category is this?",
            "How would you describe the breast composition?",
            "What composition category is assigned to the breast tissue?",
            "Which density category does this mammogram fall into?",
        ],
        answer_space=BREAST_COMPOSITION_ORDER,
        difficulty="medium",
        scope="sample",
        stage=STAGE_COMPOSITION,
        answer_source="image",
        negative_response_templates={
            "wrong_modality": "Breast composition is a mammographic assessment; it is not reported for this modality."
        },
    ),
    FieldSpec(
        field_id="breast.fgt",
        task_type="mri_fgt",
        templates=[
            "What is the amount of fibroglandular tissue?",
            "How much fibroglandular tissue is there?",
            "Which FGT category applies?",
            "Fibroglandular tissue?",
            "How is the fibroglandular tissue classified?",
            "What FGT category is reported?",
            "How would you describe the amount of fibroglandular tissue?",
            "Which fibroglandular tissue group does this case fall into?",
        ],
        answer_space=MRI_FGT_ORDER,
        difficulty="medium",
        scope="sample",
        stage=STAGE_COMPOSITION,
        answer_source="image",
        negative_response_templates={
            "wrong_modality": "Fibroglandular tissue is assessed on MRI; it is not reported for this modality."
        },
    ),
    FieldSpec(
        field_id="breast.tissue composition",
        task_type="us_tissue_composition",
        templates=[
            "What is the breast tissue composition?",
            "What background echotexture is reported?",
            "How is the tissue composition classified?",
            "Tissue composition?",
            "Which tissue composition category applies here?",
            "How would you describe the background echotexture?",
            "What composition is noted for the breast tissue?",
            "Which tissue composition group does this case fall into?",
        ],
        answer_space=TISSUE_COMPOSITION_VALUES,
        difficulty="medium",
        scope="sample",
        stage=STAGE_COMPOSITION,
        answer_source="image",
        negative_response_templates={
            "wrong_modality": "Background echotexture is an ultrasound assessment; it is not reported for this modality."
        },
    ),
    # -------------------------------------------------------------- presence
    FieldSpec(
        field_id="lesion.presence",
        task_type="lesion_presence",
        templates=[
            "Are any findings present?",
            "Is there any lesion present?",
            "Does this image show any findings?",
            "Any lesions?",
            "Are there any findings in this image?",
            "Is a lesion identified here?",
            "Do you see any lesion?",
            "Does the image contain a lesion?",
        ],
        answer_space=("yes", "no"),
        difficulty="medium",
        scope="sample",
        stage=STAGE_PRESENCE,
        answer_source="image",
    ),
    FieldSpec(
        field_id="lesion.count",
        task_type="lesion_count",
        templates=[
            "How many findings are present?",
            "How many lesions are there?",
            "Number of findings?",
            "How many distinct findings can you identify?",
            "Count the findings in this image.",
        ],
        # Free numeric: the answer is a small integer, not a term.
        answer_space=None,
        difficulty="hard",
        scope="sample",
        stage=STAGE_COUNT,
        answer_source="image",
        negative_response_templates={
            "no_lesion": "There are no findings in this image.",
        },
    ),
    # ---------------------------------------------------------- finding type
    FieldSpec(
        field_id="lesion.finding type",
        task_type="lesion_type",
        templates=[
            "What type of finding is {referent}?",
            "What is the finding type of {referent}?",
            "How is {referent} categorized?",
            "What kind of finding is {referent}?",
            "Which finding category does {referent} belong to?",
            "How would you classify {referent}?",
        ],
        solo_templates=[
            "What type of finding is present?",
            "What is the finding type?",
            "How is the finding categorized?",
            "What kind of finding is seen?",
            "Which finding category does this fall into?",
            "How would you classify this finding?",
        ],
        answer_space={
            MAMMOGRAPHY: (
                MASS,
                CALCIFICATION,
                "architectural distortion",
                ASYMMETRY,
                "lymph node",
                "skin lesion",
                "dilated duct",
            ),
            ULTRASOUND: (MASS, NON_MASS_LESION, CALCIFICATION, "lymph node"),
            MRI: (MASS, NON_MASS_ENHANCEMENT, "lymph node"),
        },
        difficulty="medium",
        scope="lesion",
        stage=STAGE_FINDING_TYPE,
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no finding, finding type is not applicable.",
        },
    ),
    # -------------------------------------------------------------- location
    FieldSpec(
        field_id="lesion.location.laterality",
        task_type="lesion_laterality",
        templates=[
            "Which breast contains {referent}?",
            "What is the laterality of {referent}?",
            "Is {referent} in the left or right breast?",
            "What side is {referent} on?",
            "Which side contains {referent}?",
            "On which side is {referent} identified?",
        ],
        answer_space=("left", "right"),
        difficulty="medium",
        scope="lesion",
        stage=STAGE_LOCATION,
        answer_source="image+context",
        requires_context=(CTX_LATERALITY,),
        negative_response_templates={
            "no_lesion": "Since there is no finding, laterality is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.location.quadrant",
        task_type="lesion_quadrant",
        templates=[
            "Which quadrant contains {referent}?",
            "What quadrant does {referent} occupy?",
            "In which breast quadrant is {referent}?",
            "What is the quadrant location of {referent}?",
            "Which breast quadrant does {referent} involve?",
        ],
        answer_space=QUADRANTS,
        difficulty="medium",
        scope="lesion",
        stage=STAGE_LOCATION,
        answer_source="image+context",
        requires_context=(CTX_LATERALITY,),
        negative_response_templates={
            "no_lesion": "Since there is no finding, location is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.location.clock position",
        task_type="lesion_clock_position",
        templates=[
            "What is the clock position of {referent}?",
            "At what o'clock position is {referent}?",
            "Where is {referent} on the clock face?",
            "At which hour is {referent} located?",
            "What is the clock-face location of {referent}?",
        ],
        answer_space=tuple(str(hour) for hour in range(1, 13)),
        difficulty="hard",
        scope="lesion",
        stage=STAGE_LOCATION,
        answer_source="image+context",
        requires_context=(CTX_LATERALITY,),
        negative_response_templates={
            "no_lesion": "Since there is no finding, clock position is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.location.depth",
        task_type="lesion_depth",
        templates=[
            "At what depth is {referent}?",
            "What is the depth of {referent}?",
            "Which third of the breast contains {referent}?",
            "How deep in the breast is {referent}?",
            "Is {referent} anterior, middle or posterior?",
        ],
        answer_space=DEPTHS,
        difficulty="hard",
        scope="lesion",
        stage=STAGE_LOCATION,
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no finding, depth is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.location.special location",
        task_type="lesion_special_location",
        templates=[
            "Is {referent} in a special location?",
            "What special location does {referent} occupy?",
            "Is {referent} retroareolar, central or in the axillary tail?",
            "Which special location applies to {referent}?",
            "What named region contains {referent}?",
        ],
        answer_space=SPECIAL_LOCATIONS,
        difficulty="medium",
        scope="lesion",
        stage=STAGE_LOCATION,
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no finding, location is not applicable.",
        },
    ),
    # ----------------------------------------------------------- mass detail
    FieldSpec(
        field_id="lesion.mass details.shape",
        task_type="mass_shape",
        templates=[
            "What is the shape of {referent}?",
            "How is the shape of {referent} described?",
            "What shape does {referent} have?",
            "Describe the shape of {referent}.",
            "Which shape category applies to {referent}?",
        ],
        answer_space=MASS_SHAPE_ORDER,
        difficulty="hard",
        scope="lesion",
        stage=STAGE_DESCRIPTOR,
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no mass, shape is not applicable.",
            "wrong_lesion_type": "Since this finding is not a mass, mass shape is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.mass details.margin",
        task_type="mass_margin",
        templates=[
            "What is the margin of {referent}?",
            "How is the margin of {referent} described?",
            "What are the margins of {referent}?",
            "How would you describe the margin of {referent}?",
            "Which margin category applies to {referent}?",
        ],
        answer_space=MARGIN_ANSWER_SPACE,
        difficulty="hard",
        scope="lesion",
        stage=STAGE_DESCRIPTOR,
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no mass, margin is not applicable.",
            "wrong_lesion_type": "Since this finding is not a mass, mass margin is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.mass details.density",
        task_type="mass_density",
        templates=[
            "What is the density of {referent}?",
            "How is the density of {referent} described?",
            "What density does {referent} have?",
            "Which density class applies to {referent}?",
            "How dense is {referent}?",
        ],
        answer_space=MASS_DENSITY_ORDER,
        difficulty="hard",
        scope="lesion",
        stage=STAGE_DESCRIPTOR,
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no mass, density is not applicable.",
            "wrong_modality": "Mass density is a mammographic descriptor; it is not reported for this modality.",
            "wrong_lesion_type": "Since this finding is not a mass, mass density is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.mass details.echo pattern",
        task_type="mass_echo_pattern",
        templates=[
            "What is the echo pattern of {referent}?",
            "How is the echogenicity of {referent} described?",
            "What echogenicity does {referent} have?",
            "Which echo pattern applies to {referent}?",
            "How echogenic is {referent}?",
        ],
        answer_space=MASS_ECHO_PATTERN_ORDER,
        difficulty="hard",
        scope="lesion",
        stage=STAGE_DESCRIPTOR,
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no mass, echo pattern is not applicable.",
            "wrong_modality": "Echo pattern is an ultrasound descriptor; it is not reported for this modality.",
            "wrong_lesion_type": "Since this finding is not a mass, mass echo pattern is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.mass details.posterior features",
        task_type="mass_posterior_features",
        templates=[
            "What are the posterior acoustic features of {referent}?",
            "How are the posterior features of {referent} described?",
            "What posterior features does {referent} show?",
            "What is seen posterior to {referent}?",
            "Which posterior acoustic pattern applies to {referent}?",
        ],
        answer_space=POSTERIOR_FEATURES_ORDER,
        difficulty="hard",
        scope="lesion",
        stage=STAGE_DESCRIPTOR,
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no mass, posterior features are not applicable.",
            "wrong_modality": "Posterior features are an ultrasound descriptor; they are not reported for this modality.",
            "wrong_lesion_type": "Since this finding is not a mass, posterior features are not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.mass details.echogenic feature",
        task_type="mass_echogenic_feature",
        templates=[
            "What echogenic boundary feature does {referent} show?",
            "Does {referent} have an echogenic pseudocapsule or an echogenic rind?",
            "How is the echogenic boundary of {referent} described?",
            "Which echogenic feature applies to {referent}?",
        ],
        answer_space=ECHOGENIC_FEATURES,
        difficulty="hard",
        scope="lesion",
        stage=STAGE_DESCRIPTOR,
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no mass, echogenic features are not applicable.",
            "wrong_modality": "Echogenic pseudocapsule and rind are ultrasound descriptors; they are not reported for this modality.",
            "wrong_lesion_type": "Since this finding is not a mass, echogenic features are not applicable.",
        },
    ),
    # -------------------------------------------------- calcification detail
    FieldSpec(
        field_id="lesion.calcification details.type",
        task_type="calcification_type_family",
        templates=[
            "What calcification type category does {referent} fall into?",
            "How would you categorize {referent}?",
            "Is {referent} typically benign or suspicious in morphology?",
            "Which calcification family applies to {referent}?",
        ],
        answer_space=CALCIFICATION_TYPE_FAMILIES,
        difficulty="hard",
        scope="lesion",
        stage=STAGE_DESCRIPTOR,
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there are no calcifications, calcification type is not applicable.",
            "wrong_modality": "Calcification morphology is a mammographic descriptor; it is not reported for this modality.",
            "wrong_lesion_type": "Since this finding is not a group of calcifications, calcification type is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.calcification details.type details",
        task_type="calcification_type_details",
        templates=[
            "What is the calcification morphology of {referent}?",
            "How is the morphology of {referent} described?",
            "What morphology does {referent} show?",
            "Which morphology category applies to {referent}?",
        ],
        answer_space=CALCIFICATION_MORPHOLOGIES,
        difficulty="hard",
        scope="lesion",
        stage=STAGE_DESCRIPTOR,
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there are no calcifications, morphology is not applicable.",
            "wrong_modality": "Calcification morphology is a mammographic descriptor; it is not reported for this modality.",
            "wrong_lesion_type": "Since this finding is not a group of calcifications, morphology is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.calcification details.distribution",
        task_type="calcification_distribution",
        templates=[
            "What is the distribution of {referent}?",
            "How is {referent} distributed?",
            "What distribution pattern does {referent} show?",
            "Which distribution class applies to {referent}?",
        ],
        answer_space=CALCIFICATION_DISTRIBUTION_ORDER,
        difficulty="hard",
        scope="lesion",
        stage=STAGE_DESCRIPTOR,
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there are no calcifications, distribution is not applicable.",
            "wrong_modality": "Calcification distribution is a mammographic descriptor; it is not reported for this modality.",
            "wrong_lesion_type": "Since this finding is not a group of calcifications, distribution is not applicable.",
        },
    ),
    # ------------------------------------------------------------ assessment
    FieldSpec(
        field_id="assessment.birads",
        task_type="assessment_birads",
        templates=[
            "What is the BI-RADS assessment?",
            "What BI-RADS category applies to this case?",
            "What is the final BI-RADS classification?",
            "BI-RADS?",
            "Which BI-RADS assessment is assigned?",
            "How is this case categorized in BI-RADS?",
            "What is the overall BI-RADS result?",
        ],
        answer_space=BIRADS_ANSWER_SPACE,
        difficulty="hard",
        scope="sample",
        stage=STAGE_ASSESSMENT,
        answer_source="image+context",
    ),
]

FIELD_SPECS_BY_ID: dict[str, FieldSpec] = {spec.field_id: spec for spec in FIELD_SPECS}


# ===========================================================================
# APPLICABILITY RULES
# ===========================================================================
MODALITY_RULES: dict[str, set[str]] = {
    "breast.composition": {MAMMOGRAPHY},
    "breast.fgt": {MRI},
    "breast.tissue composition": {ULTRASOUND},
    "lesion.mass details.density": {MAMMOGRAPHY},
    "lesion.mass details.echo pattern": {ULTRASOUND},
    "lesion.mass details.posterior features": {ULTRASOUND},
    "lesion.mass details.echogenic feature": {ULTRASOUND},
    "lesion.calcification details.type": {MAMMOGRAPHY},
    "lesion.calcification details.type details": {MAMMOGRAPHY},
    "lesion.calcification details.distribution": {MAMMOGRAPHY},
}

LESION_TYPE_RULES: dict[str, set[str]] = {
    "lesion.mass details.shape": {MASS},
    "lesion.mass details.margin": {MASS},
    "lesion.mass details.density": {MASS},
    "lesion.mass details.echo pattern": {MASS},
    "lesion.mass details.posterior features": {MASS, NON_MASS_LESION},
    "lesion.mass details.echogenic feature": {MASS},
    "lesion.calcification details.type": {CALCIFICATION},
    "lesion.calcification details.type details": {CALCIFICATION},
    "lesion.calcification details.distribution": {CALCIFICATION},
}

# field_specs.py, after LESION_TYPE_RULES
ANY_FINDING_FIELDS: frozenset[str] = frozenset(
    field_id
    for field_id, spec in FIELD_SPECS_BY_ID.items()
    if field_id not in LESION_TYPE_RULES
    and "no_lesion" in (spec.negative_response_templates or {})
)


def validate_specs() -> None:
    """Guard the invariants the generator relies on. Called at import time."""
    problems = []
    for field_id, spec in FIELD_SPECS_BY_ID.items():
        allowed = MODALITY_RULES.get(field_id)
        if allowed and isinstance(spec.answer_space, dict):
            missing = allowed - set(spec.answer_space)
            if missing:
                problems.append(
                    f"{field_id}: no answer space for modality {sorted(missing)}"
                )
        if spec.scope == "lesion" and "{referent}" not in " ".join(spec.templates):
            if field_id not in {"lesion.count"}:
                problems.append(
                    f"{field_id}: per-finding templates need a {{referent}}"
                )
        if spec.solo_templates and "{referent}" in " ".join(spec.solo_templates):
            problems.append(f"{field_id}: solo templates must not take a {{referent}}")
    for field_id in set(MODALITY_RULES) | set(LESION_TYPE_RULES):
        if field_id not in FIELD_SPECS_BY_ID:
            problems.append(f"{field_id}: rule refers to a field with no FieldSpec")
    if problems:
        raise AssertionError("Invalid field specs:\n  " + "\n  ".join(problems))


validate_specs()


# ===========================================================================
# NEGATIVE (INAPPLICABLE-QUESTION) CANDIDATES
# ===========================================================================
@dataclass(frozen=True)
class NegativeCandidate:
    field_id: str
    case: NegativeCase
    answer: str
    referent: str | None = None


def get_negative_candidates(
    field_specs_by_id: dict[str, FieldSpec],
    modality: str,
    lesion_status: str,
    lesion_types: tuple[str | None, ...] = (),
    annotation_scope: str = "unknown",
) -> list[NegativeCandidate]:
    candidates: list[NegativeCandidate] = []
    present_types = set(lesion_types)

    for field_id, spec in field_specs_by_id.items():
        neg_templates = spec.negative_response_templates
        if not neg_templates:
            continue

        reasons: list[NegativeCase] = []

        allowed_modalities = MODALITY_RULES.get(field_id)
        if (
            allowed_modalities is not None
            and modality not in allowed_modalities
            and "wrong_modality" in neg_templates
        ):
            reasons.append("wrong_modality")

        allowed_lesion_types = LESION_TYPE_RULES.get(field_id)
        if lesion_status == "absent":
            if (
                allowed_lesion_types is not None or field_id in ANY_FINDING_FIELDS
            ) and "no_lesion" in neg_templates:
                reasons.append("no_lesion")
        elif (
            lesion_status == "present"
            and annotation_scope == "complete"
            and lesion_types
            and all(lesion_types)
            and allowed_lesion_types is not None
        ):
            if not (present_types & allowed_lesion_types):
                if "wrong_lesion_type" in neg_templates:
                    reasons.append("wrong_lesion_type")
                elif "no_lesion" in neg_templates:
                    reasons.append("no_lesion")

        for case in reasons:
            candidates.append(
                NegativeCandidate(
                    field_id=field_id, case=case, answer=neg_templates[case]
                )
            )

    return candidates


def sample_negative_candidates_one_per_field(
    rng,
    candidates: list[NegativeCandidate],
    k: int,
) -> list[NegativeCandidate]:
    by_field: dict[str, list[NegativeCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_field[candidate.field_id].append(candidate)

    field_ids = sorted(by_field)
    if not field_ids:
        return []

    chosen_fields = rng.sample(field_ids, k=min(k, len(field_ids)))
    return [rng.choice(by_field[field_id]) for field_id in chosen_fields]
