import logging
import os
import re
from functools import partial
from glob import glob

import pandas as pd
import pydicom as dicom

from dtos.breast_cancer_dataset import (
    ASSESSMENT_EXAM_LEVEL,
    BreastCancerDataset,
    ExamInformation,
    stable_exam_id,
)
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext, PatientContext
from dtos.json_report_findings import (
    SCOPE_COMPLETE,
    Assessment,
    BreastFindings,
    CalcificationAbnormality,
    JSONReportFindings,
    Lesion,
    LesionLocation,
    MassAbnormality,
)
from utils.error_handling import trycatch_func
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    BIRADS_MAPPING,
    CALCIFICATION,
    MAMMOGRAPHY,
    MASS,
    NOT_PRESENT,
    abbreviation_mapping,
    bin_age,
    convert_dcm_image,
    get_proper_birads,
    get_proper_breast_density,
    get_proper_calcification_distribution,
    get_proper_calcification_type,
    get_proper_exam_view,
    get_proper_finding_type,
    get_proper_location,
    get_proper_mass_density,
    get_value,
    isna_v2,
    map_birads,
    migrate_mammography_microlobulated,
    normalize_laterality,
    region_mapping,
)

raw_imgs_path = "../data/raw/tompei-cmmd/cmmd"
raw_imgs_extension = ".dcm"
excel_path = "../data/raw/tompei-cmmd/TOMPEI-CMMD_clinical_data_v01_20250121.xlsx"
processed_imgs_path = "../data/processed/tompei-cmmd/imgs"
csv_save_path = "../data/processed/tompei-cmmd/tompei-cmmd.csv"


class TompeiCMMD(BreastCancerDataset):
    _modality = MAMMOGRAPHY
    _machine = "GE Senographe DS mammography system"
    _race = "asian"

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(
                modality=MAMMOGRAPHY,
                read_func=partial(convert_dcm_image),
            ),
        )

        self.clinical_data = pd.read_excel(
            excel_path, sheet_name="Imaging Diagnosis Details Sheet"
        )

        adjusted_cols = [
            "patient id",
            "laterality",
            "age",
            "classification",
            "exclusion reason",
            "breast density",
            "mass location",
            "mass shape",
            "mass margin",
            "mass density",
            "mass associated calcification",
            "mass other associated findings",
            "calcification location",
            "calcification morphology",
            "calcification distribution",
            "calcification clearly benign calcifications",
            "other associated findings",
            "other findings location",
            "other findings breast parenchyma",
            "other findings skin",
            "other findings lymph nodes",
            "birads",
            "notes",
            "mass - additional lesion location",
            "mass - additional lesion shape",
            "mass - additional lesion margin",
            "mass - additional lesion density",
            "mass - additional lesion associated calcification",
            "mass - additional lesion other associated findings",
            "calcification - additional lesion location",
            "calcification - additional lesion morphology",
            "calcification - additional lesion distribution",
            "calcification - additional lesion clearly benign calcifications",
        ]

        self.clinical_data = self.clinical_data.iloc[1:]
        self.clinical_data.columns = adjusted_cols

    def process_info(self):
        self.report.rows_seen = len(self.clinical_data)
        # Exclude rows with exclusion reasons
        self.clinical_data = self.clinical_data[
            self.clinical_data["exclusion reason"].isna()
        ]

        # Remove rows with nan birads
        self.clinical_data = self.clinical_data[self.clinical_data["birads"].notna()]
        self.clinical_data["birads"] = self.clinical_data["birads"].apply(
            lambda x: int(x)
        )

        # Adjust processing for clarity
        self.clinical_data["laterality"] = self.clinical_data["laterality"].apply(
            normalize_laterality
        )
        self.clinical_data["birads"] = self.clinical_data["birads"].apply(
            lambda x: map_birads(x, source="TOMPEI-CMMD", modality=MAMMOGRAPHY)
        )
        self.clinical_data = self.clinical_data[self.clinical_data["birads"].notna()]
        self.clinical_data["mass location"] = self.clinical_data["mass location"].apply(
            lambda x: get_value(x, region_mapping) if not isna_v2(x) else None
        )
        self.clinical_data["calcification location"] = self.clinical_data[
            "calcification location"
        ].apply(lambda x: get_value(x, region_mapping) if not isna_v2(x) else None)
        self.clinical_data["other findings location"] = self.clinical_data[
            "other findings location"
        ].apply(lambda x: get_value(x, region_mapping) if not isna_v2(x) else None)
        self.clinical_data["mass - additional lesion location"] = self.clinical_data[
            "mass - additional lesion location"
        ].apply(lambda x: get_value(x, region_mapping) if not isna_v2(x) else None)
        self.clinical_data["calcification - additional lesion location"] = (
            self.clinical_data["calcification - additional lesion location"].apply(
                lambda x: get_value(x, region_mapping) if not isna_v2(x) else None
            )
        )

        # Convert everything to lowercase for consistency
        self.clinical_data = self.clinical_data.apply(
            lambda col: (
                col.str.lower()
                if col.dtype == "object" or col.dtype.name == "string"
                else col
            )
        )

        pat = re.compile(
            r"\b(?:" + "|".join(map(re.escape, abbreviation_mapping.keys())) + r")\b"
        )
        self.clinical_data = self.clinical_data.map(
            lambda s: (
                pat.sub(lambda m: abbreviation_mapping[m.group(0)], s)
                if isinstance(s, str)
                else s
            )
        )

        # Count the number of images per patient. Drop all that dont have exactly 2 images (cc + mlo). 0 is cases for healthy patients, 2 is cases for patients with one side of the breast, 4 is for patients with both sides of the breast affected (but we dont know how to distinguish them via the metadata => the solution is to either give them all, or drop them, we will drop them for now to avoid complications)
        self.clinical_data["n_images"] = self.clinical_data["patient id"].apply(
            lambda pid: len(
                glob(
                    os.path.join(
                        raw_imgs_path, pid.upper(), "**", f"*{raw_imgs_extension}"
                    ),
                    recursive=True,
                )
            )
        )
        self.clinical_data = self.clinical_data[self.clinical_data["n_images"] == 2]

        self.run_row_pool(self.clinical_data)

    @trycatch_func
    def process_row(self, row):
        out = []

        # get patient images
        imgs = sorted(
            glob(
                os.path.join(
                    raw_imgs_path,
                    row["patient id"].upper(),
                    "**",
                    f"*{raw_imgs_extension}",
                ),
                recursive=True,
            )
        )

        for img_path in imgs:
            exam_id = stable_exam_id(
                self.get_dataset_name(), os.path.relpath(img_path, raw_imgs_path)
            )
            view = self.get_view(img_path)
            if view is None:
                logging.warning(
                    f"Could not determine the view for {img_path} (patient {row['patient id']}). Skipping image."
                )
                continue

            exam_imgs_path = self.save_imgs(img_path, exam_id, row["patient id"])

            if isna_v2(exam_imgs_path):
                logging.warning(
                    f"Skipping image for patient {row['patient id']} exam {exam_id} due to processing error."
                )
                continue

            # construct context
            context = JSONReportContext(
                patient_context=PatientContext(age=bin_age(row["age"])),
                exam_context=ExamContext(
                    modality=self._modality,
                    laterality=row["laterality"],
                    view=view,
                ),
            )

            assessment_birads = get_proper_birads(row["birads"], modality=MAMMOGRAPHY)

            findings = JSONReportFindings.build(
                breast=BreastFindings(
                    composition=get_proper_breast_density(row["breast density"])
                ),
                lesion=(
                    NOT_PRESENT
                    if row["birads"] == BIRADS_MAPPING[1]
                    else self._construct_lesions(row)
                ),
                assessment=Assessment(birads=assessment_birads),
                annotation_scope=SCOPE_COMPLETE,
            )

            current_exam = ExamInformation(
                id=exam_id,
                patient=f"{self.get_dataset_name()}-{row['patient id']}",
                dataset=self.get_dataset_name(),
                modality=self._modality,
                birads=assessment_birads,
                race=self._race,
                machine=self._machine,
                exam=exam_imgs_path,
                segmentation=None,
                context=context.get_string(),
                findings=findings.get_string(),
                exam_id=f"{self.get_dataset_name()}-{row['patient id']}",
                breast_id=f"{self.get_dataset_name()}-{row['patient id']}-{row['laterality']}",
                assessment_scope=ASSESSMENT_EXAM_LEVEL,
            )

            out.append(current_exam)

        return out

    def _has_text(self, value) -> bool:
        return not isna_v2(value) and str(value).strip() != ""

    def _build_mass_candidate(self, row, prefix: str, is_primary: bool):
        location_col = f"{prefix}location"
        shape_col = f"{prefix}shape"
        margin_col = f"{prefix}margin"
        density_col = f"{prefix}density"
        assoc_calc_col = f"{prefix}associated calcification"
        assoc_findings_col = f"{prefix}other associated findings"

        location = row.get(location_col)
        shape = row.get(shape_col)
        margin = row.get(margin_col)
        density = row.get(density_col)
        assoc_calc = row.get(assoc_calc_col)
        assoc_findings = row.get(assoc_findings_col)

        has_core_mass_info = any(
            [
                self._has_text(shape),
                self._has_text(margin),
                self._has_text(density),
            ]
        )
        if not has_core_mass_info:
            return None

        descriptor_count = sum(
            [
                self._has_text(shape),
                self._has_text(margin),
                self._has_text(density),
            ]
        )
        score = (
            (2 if is_primary else 0)
            + int(self._has_text(location))
            + descriptor_count
            + int(self._has_text(assoc_calc) or self._has_text(assoc_findings))
        )

        mapped_shape, mapped_margin = migrate_mammography_microlobulated(shape, margin)
        finding_type = (
            get_proper_finding_type(shape, MAMMOGRAPHY)
            if mapped_shape is None and self._has_text(shape)
            else None
        ) or MASS

        return {
            "score": score,
            "is_primary": is_primary,
            "descriptor_count": descriptor_count,
            "type_priority": 2,
            "lesion": Lesion(
                finding_type=finding_type,
                location=LesionLocation(
                    **get_proper_location(location, row.get("laterality"))
                ),
                mass_details=(
                    MassAbnormality(
                        shape=mapped_shape,
                        margin=mapped_margin,
                        density=get_proper_mass_density(density),
                    )
                    if finding_type == MASS
                    else None
                ),
            ),
        }

    def _build_calcification_candidate(
        self, row, prefix: str, is_primary: bool, associated_col: str | None = None
    ):
        location_col = f"{prefix}location"
        morphology_col = f"{prefix}morphology"
        distribution_col = f"{prefix}distribution"
        benign_calc_col = f"{prefix}clearly benign calcifications"

        location = row.get(location_col)
        morphology = row.get(morphology_col)
        distribution = row.get(distribution_col)
        benign_calc = row.get(benign_calc_col)
        associated = row.get(associated_col) if associated_col is not None else None

        has_core_calc_info = any(
            [
                self._has_text(morphology),
                self._has_text(distribution),
                self._has_text(benign_calc),
            ]
        )
        if not has_core_calc_info:
            return None

        descriptor_count = sum(
            [
                self._has_text(morphology),
                self._has_text(distribution),
                self._has_text(benign_calc),
            ]
        )
        score = (
            (2 if is_primary else 0)
            + int(self._has_text(location))
            + descriptor_count
            + int(self._has_text(associated))
        )

        calc_type, calc_det = get_proper_calcification_type(morphology)
        return {
            "score": score,
            "is_primary": is_primary,
            "descriptor_count": descriptor_count,
            "type_priority": 1,
            "lesion": Lesion(
                finding_type=CALCIFICATION,
                location=LesionLocation(
                    **get_proper_location(location, row.get("laterality"))
                ),
                calcification_details=CalcificationAbnormality(
                    type=calc_type,
                    type_details=calc_det,
                    distribution=get_proper_calcification_distribution(distribution),
                ),
            ),
        }

    def _construct_lesions(self, row) -> list[Lesion] | None:
        candidates = [
            self._build_mass_candidate(row, "mass ", is_primary=True),
            self._build_calcification_candidate(
                row,
                "calcification ",
                is_primary=True,
                associated_col="other associated findings",
            ),
            self._build_mass_candidate(
                row, "mass - additional lesion ", is_primary=False
            ),
            self._build_calcification_candidate(
                row, "calcification - additional lesion ", is_primary=False
            ),
        ]

        valid_candidates = [
            candidate for candidate in candidates if candidate is not None
        ]
        if not valid_candidates:
            return None

        valid_candidates.sort(
            key=lambda c: (
                c["score"],
                c["descriptor_count"],
                c["is_primary"],
                c["type_priority"],
            ),
            reverse=True,
        )

        lesions = [candidate["lesion"] for candidate in valid_candidates]
        lesions[0].is_index = True
        return lesions

    def get_view(self, img_path: str) -> str | None:
        try:
            ds = dicom.dcmread(img_path, stop_before_pixels=True)
        except Exception as e:
            logging.warning(f"Could not read DICOM header at {img_path}: {e}")
            return None

        view = getattr(ds, "ViewPosition", None)

        if not view and "ViewCodeSequence" in ds:
            try:
                view = ds.ViewCodeSequence[0].CodeMeaning
            except Exception:
                view = None

        if not view:
            return None

        return get_proper_exam_view(str(view))

    def save_imgs(self, image_path: str, exam_id: str, patient_id: str):
        res = self.image_processor.process_image(img_path=image_path)
        if res is None:
            return None

        save_path = os.path.join(
            self.image_processor.processed_imgs_path,
            f"{patient_id}-{exam_id}{self.image_processor.IMGS_SUFFIX}",
        )
        return self.image_processor.save_process(save_path, [res])
