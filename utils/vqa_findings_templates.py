from collections import defaultdict
from dataclasses import dataclass
from typing import Literal, Optional

AnswerSource = Literal["image", "image+context"]
NegativeCase = Literal["no_lesion", "wrong_modality", "wrong_lesion_type"]


@dataclass
class FieldSpec:
    field_id: str
    task_type: str
    templates: list[str]
    difficulty: str = "medium"
    scope: str = "sample"  # sample or lesion
    answer_source: AnswerSource = "image"
    negative_response_templates: Optional[dict[NegativeCase, str]] = None  # mapping from case to list of templates


FIELD_SPECS: list[FieldSpec] = [
    FieldSpec(
        field_id="assessment.birads",
        task_type="assessment_birads",
        templates=[
            "What is the BI-RADS assessment?",
            "What BI-RADS category applies to this case?",
            "What is the final BI-RADS classification?",
            "BI-RADS?",
            "What is the final BI-RADS category?",
            "Which BI-RADS assessment is assigned?",
            "How is this case categorized in BI-RADS?",
            "What BI-RADS score is reported?",
            "What is the overall BI-RADS result?",
            "Which BI-RADS class does this case fall under?",
        ],
        difficulty="hard",
        scope="sample",
        answer_source="image+context",
    ),
    FieldSpec(
        field_id="breast.density",
        task_type="breast_density",
        templates=[
            "What is the breast density?",
            "What breast density category is reported?",
            "How is the breast density classified?",
            "Breast density?",
            "Which breast density category applies here?",
            "What density category is assigned to the breast tissue?",
            "How would you describe the breast density?",
            "What is the reported density classification?",
            "What density is noted for the breasts?",
            "Which density group does this case belong to?",
        ],
        difficulty="medium",
        scope="sample",
        answer_source="image+context",  # yes, implants case
        negative_response_templates={"wrong_modality": "Breast density is not assessed on this modality. Only on MG and MR."},
    ),
    FieldSpec(
        field_id="breast.tissue composition",
        task_type="us_tissue_composition",
        templates=[
            "What is the breast tissue composition?",
            "What tissue composition is reported?",
            "How is the tissue composition classified?",
            "Tissue composition?",
            "Which tissue composition category applies here?",
            "How would you describe the breast tissue composition?",
            "What composition is noted for the breast tissue?",
            "What is the reported tissue composition category?",
            "How is the US tissue composition described?",
            "Which tissue composition group does this case fall into?",
        ],
        difficulty="medium",
        scope="sample",
        answer_source="image+context",
        negative_response_templates={"wrong_modality": "Tissue composition is not assessed on this modality. Only on US."},
    ),
    FieldSpec(
        field_id="lesion.presence",
        task_type="lesion_presence",
        templates=[
            "Are any lesions present?",
            "Is there any lesion present?",
            "Does this case show any lesions?",
            "Any lesions?",
            "Are there any lesions in this case?",
            "Is a lesion identified here?",
            "Do you see any lesion?",
            "Are any suspicious lesions present?",
            "Is any lesion present?",
            "Does the image contain a lesion?",
        ],
        difficulty="medium",
        scope="sample",
        answer_source="image",
    ),
    FieldSpec(
        field_id="lesion.type",
        task_type="lesion_type",
        templates=[
            "What type of lesion is present?",
            "What is the lesion type?",
            "What kind of lesion is this?",
            "Lesion type?",
            "How is the lesion categorized?",
            "What lesion category is shown?",
            "What sort of lesion is present?",
            "What abnormality type is this lesion?",
            "Which type of finding is this lesion?",
            "How would you classify the lesion type?",
        ],
        difficulty="medium",
        scope="lesion",
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no lesion, lesion type is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.location",
        task_type="lesion_location",
        templates=[
            "Where is the lesion located?",
            "What is the lesion location?",
            "Where is this lesion?",
            "What location is reported for the lesion?",
            "In what location is the lesion found?",
            "Where is the finding located?",
            "What is the position of the lesion?",
            "Where can the lesion be seen?",
            "What site does the lesion involve?",
            "Where in the breast is the lesion located?",
        ],
        difficulty="medium",
        scope="lesion",
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no lesion, location is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.laterality",
        task_type="lesion_laterality",
        templates=[
            "What is the lesion laterality?",
            "Which breast contains the lesion?",
            "Is the lesion in the left or right breast?",
            "What side is the lesion on?",
            "Which side contains the lesion?",
            "What is the side of the lesion?",
            "Is the lesion located in the left breast or the right breast?",
            "Which breast is involved by the lesion?",
            "What laterality is reported for the lesion?",
            "On which side is the lesion identified?",
        ],
        difficulty="medium",
        scope="lesion",
        answer_source="image+context",
        negative_response_templates={
            "no_lesion": "Since there is no lesion, laterality is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.mass details.shape",
        task_type="mass_shape",
        templates=[
            "What is the mass shape?",
            "How is the mass shape described?",
            "What shape is the mass?",
            "Shape?",
            "What shape does the lesion have?",
            "Describe the lesion shape.",
            "How would you describe the mass shape?",
            "What is the shape of this mass?",
            "Which shape category applies to the mass?",
            "How is this mass shaped?",
        ],
        difficulty="hard",
        scope="lesion",
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no mass, shape is not applicable.",
            "wrong_lesion_type": "Since this lesion is not a mass, mass shape is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.mass details.margin",
        task_type="mass_margin",
        templates=[
            "What is the mass margin?",
            "How is the mass margin described?",
            "Mass margin?",
            "What are the margins of the mass?",
            "How would you describe the mass margin?",
            "What margin type is reported for the mass?",
            "How are the mass borders described?",
            "What is the border characteristic of the mass?",
            "Which margin category applies to the mass?",
            "What is the margin appearance of the mass?",
        ],
        difficulty="hard",
        scope="lesion",
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no mass, margin is not applicable.",
            "wrong_lesion_type": "Since this lesion is not a mass, mass margin is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.mass details.density",
        task_type="mass_density",
        templates=[
            "What is the mass density?",
            "How is the mass density described?",
            "Mass density?",
            "What density does the mass have?",
            "How would you describe the density of the mass?",
            "What density category is reported for the mass?",
            "How is the lesion density characterized?",
            "What is the reported density of this mass?",
            "Which density class applies to the mass?",
            "How dense is the mass?",
        ],
        difficulty="hard",
        scope="lesion",
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no mass, density is not applicable.",
            "wrong_modality": "Mass density is not applicable to this modality. Only to MG and MR.",
            "wrong_lesion_type": "Since this lesion is not a mass, mass density is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.mass details.echogenicity",
        task_type="mass_echogenicity",
        templates=[
            "What is the mass echogenicity?",
            "How is the mass echogenicity described?",
            "Mass echogenicity?",
            "What echogenicity does the mass have?",
            "How would you describe the echogenicity of the mass?",
            "What echogenicity category is reported?",
            "How is the lesion echogenicity characterized?",
            "What is the reported echogenicity of the mass?",
            "Which echogenicity class applies to the mass?",
            "How echogenic is the mass?",
        ],
        difficulty="hard",
        scope="lesion",
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no mass, echogenicity is not applicable.",
            "wrong_modality": "Mass echogenicity is not applicable to this modality. Only to US.",
            "wrong_lesion_type": "Since this lesion is not a mass, mass echogenicity is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.mass details.posterior features",
        task_type="mass_posterior_features",
        templates=[
            "What are the posterior acoustic features?",
            "How are the posterior features described?",
            "Posterior acoustic features?",
            "What posterior features are present?",
            "How would you describe the posterior acoustic behavior?",
            "What posterior acoustic pattern is reported?",
            "How is the posterior effect characterized?",
            "What is noted posterior to the mass?",
            "Which posterior acoustic features apply here?",
            "What are the posterior acoustic findings?",
        ],
        difficulty="hard",
        scope="lesion",
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no mass, posterior features is not applicable.",
            "wrong_modality": "Mass posterior features is not applicable to this modality. Only to US.",
            "wrong_lesion_type": "Since this lesion is not a mass, mass posterior features is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.mass details.halo",
        task_type="mass_halo",
        templates=[
            "Is a halo present around the mass?",
            "Is there a halo sign?",
            "Halo present?",
            "Is a halo seen around the lesion?",
            "Does the mass show a halo?",
            "Is a surrounding halo identified?",
            "Is there a halo around the mass?",
            "Do you see a halo sign around the lesion?",
            "Is halo formation present?",
            "Is a peripheral halo present?",
        ],
        difficulty="hard",
        scope="lesion",
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no mass, halo is not applicable.",
            "wrong_modality": "Mass halo is not applicable to this modality. Only to US.",
            "wrong_lesion_type": "Since this lesion is not a mass, mass halo is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.calcification details.type",
        task_type="calcification_type_family",
        templates=[
            "What is the calcification type category?",
            "How are the calcifications categorized?",
            "Calcification type?",
            "What type of calcifications are present?",
            "How would you classify the calcification type?",
            "What calcification category is reported?",
            "Which calcification family applies here?",
            "What is the reported calcification class?",
            "How is the calcification pattern categorized?",
            "What type category do the calcifications fall into?",
        ],
        difficulty="hard",
        scope="lesion",
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there is no lesion, calcification type is not applicable.",
            "wrong_modality": "Calcification type is not applicable to this modality. Only to MG and MR.",
            "wrong_lesion_type": "Since this lesion is not a calcification, calcification type is not applicable.",
        },
    ),
    FieldSpec(
        field_id="lesion.calcification details.type details",
        task_type="calcification_type_details",
        templates=[
            "What is the calcification morphology?",
            "How are the calcifications morphologically described?",
            "Calcification morphology?",
            "What morphology do the calcifications have?",
            "How would you describe the calcification morphology?",
            "What morphologic pattern is reported for the calcifications?",
            "How is the calcification appearance characterized?",
            "What is the detailed morphology of the calcifications?",
            "Which morphology category applies to the calcifications?",
            "What is the reported calcification form?",
        ],
        difficulty="hard",
        scope="lesion",
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there are no calcifications, morphology is not applicable.",
            "wrong_modality": "Calcification morphology is not applicable to this modality. Only to MG and MR.",
        },
    ),
    FieldSpec(
        field_id="lesion.calcification details.distribution",
        task_type="calcification_distribution",
        templates=[
            "What is the calcification distribution?",
            "How are the calcifications distributed?",
            "Calcification distribution?",
            "What distribution pattern do the calcifications show?",
            "How would you describe the calcification distribution?",
            "What distribution category is reported for the calcifications?",
            "How is the spread of the calcifications characterized?",
            "What is the reported calcification distribution pattern?",
            "Which distribution class applies to the calcifications?",
            "How are these calcifications arranged?",
        ],
        difficulty="hard",
        scope="lesion",
        answer_source="image",
        negative_response_templates={
            "no_lesion": "Since there are no calcifications, distribution is not applicable.",
            "wrong_modality": "Calcification distribution is not applicable to this modality. Only to MG and MR.",
        },
    ),
]

MODALITY_RULES: dict[str, set[str]] = {
    "breast.density": {"mg", "mr"},
    "breast.tissue composition": {"us"},
    "lesion.mass details.echogenicity": {"us"},
    "lesion.mass details.posterior features": {"us"},
    "lesion.mass details.halo": {"us"},
    "lesion.mass details.density": {"mg", "mr"},
    "lesion.calcification details.type": {"mg", "mr"},
    "lesion.calcification details.type details": {"mg", "mr"},
    "lesion.calcification details.distribution": {"mg", "mr"},
}

LESION_TYPE_RULES: dict[str, set[str]] = {
    "lesion.mass details.shape": {"mass"},
    "lesion.mass details.margin": {"mass"},
    "lesion.mass details.density": {"mass"},
    "lesion.mass details.echogenicity": {"mass"},
    "lesion.mass details.posterior features": {"mass"},
    "lesion.mass details.halo": {"mass"},
    "lesion.calcification details.type": {"calcification"},
    "lesion.calcification details.type details": {"calcification"},
    "lesion.calcification details.distribution": {"calcification"},
}


@dataclass(frozen=True)
class NegativeCandidate:
    field_id: str
    case: NegativeCase
    answer: str


def get_negative_candidates(
    field_specs_by_id: dict[str, FieldSpec],
    modality: str,
    lesions_present: int,  # 0 for unknown (in which we do not ask lesion questions), 1 for present, -1 for absent
    lesion_type: str,
) -> list[NegativeCandidate]:
    candidates: list[NegativeCandidate] = []

    for field_id, fs in field_specs_by_id.items():
        neg_templates = fs.negative_response_templates
        if not neg_templates:
            continue

        reasons: list[NegativeCase] = []

        # 1) wrong modality
        allowed_modalities = MODALITY_RULES.get(field_id)
        if allowed_modalities is not None and modality not in allowed_modalities and "wrong_modality" in neg_templates:
            reasons.append("wrong_modality")

        # 2) lesion-specific inapplicability
        allowed_lesion_types = LESION_TYPE_RULES.get(field_id)
        if allowed_lesion_types is not None:
            if lesions_present == 0:
                continue  # if lesion status is unknown, we do not add negative candidates for lesion-specific fields, as we don't know if they are applicable or not
            if lesions_present == -1:
                if "no_lesion" in neg_templates:
                    reasons.append("no_lesion")
            else:
                has_matching_lesion = lesion_type in allowed_lesion_types
                if not has_matching_lesion:
                    if "wrong_lesion_type" in neg_templates:
                        reasons.append("wrong_lesion_type")
                    elif "no_lesion" in neg_templates:
                        reasons.append("no_lesion")

        for case in reasons:
            candidates.append(
                NegativeCandidate(
                    field_id=field_id,
                    case=case,
                    answer=neg_templates[case],
                )
            )

    return candidates


def sample_negative_candidates_one_per_field(
    rng,
    candidates: list[NegativeCandidate],
    k: int,
) -> list[NegativeCandidate]:
    by_field: dict[str, list[NegativeCandidate]] = defaultdict(list)
    for c in candidates:
        by_field[c.field_id].append(c)

    field_ids = list(by_field)
    if not field_ids:
        return []

    chosen_fields = rng.sample(field_ids, k=min(k, len(field_ids)))

    sampled: list[NegativeCandidate] = []
    for field_id in chosen_fields:
        sampled.append(rng.choice(by_field[field_id]))

    return sampled
