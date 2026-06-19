import logging
import os
import re
import uuid
from functools import partial
from glob import glob
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from dtos.breast_cancer_dataset import (
    BreastCancerDataset,
    ExamInformation,
)
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext, PatientContext
from dtos.json_report_findings import (
    Assessment,
    BreastFindings,
    CalcificationAbnormality,
    JSONReportFindings,
    Lesion,
    MassAbnormality,
)
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    CALCIFICATION,
    CRANIAL_CAUDAL,
    MASS,
    MEDIOLATERAL_OBLIQUE,
    NOT_PRESENT,
    abbreviation_mapping,
    bin_age,
    birads_assessment,
    birads_mapping,
    convert_dcm_image,
    get_proper_birads,
    get_proper_breast_density,
    get_proper_calcification_distribution,
    get_proper_calcification_type,
    get_proper_location,
    get_proper_mass_density,
    get_proper_mass_margin,
    get_proper_mass_shape,
    get_value,
    isna_v2,
    laterality,
    region_mapping,
    sanitize_age,
)

raw_imgs_path = "../data/raw/tompei-cmmd/cmmd"
raw_imgs_extension = ".dcm"
excel_path = "../data/raw/tompei-cmmd/TOMPEI-CMMD_clinical_data_v01_20250121.xlsx"
processed_imgs_path = "../data/processed/tompei-cmmd/imgs"
csv_save_path = "../data/processed/tompei-cmmd/tompei-cmmd.csv"


class TompeiCMMD(BreastCancerDataset):
    _modality = "mg"
    _machine = "GE Senographe DS mammography system"
    _race = "asian"

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(
                read_func=partial(convert_dcm_image)
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
            lambda x: get_value(x, laterality) if not isna_v2(x) else None
        )
        self.clinical_data["birads"] = self.clinical_data["birads"].apply(
            lambda x: get_value(x, birads_assessment)
        )
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

        # Process the exams
        n = cpu_count() - 1
        df_split = [
            self.clinical_data.iloc[idx]
            for idx in np.array_split(np.arange(len(self.clinical_data)), n)
            if len(idx) > 0
        ]
        with Pool(processes=n) as p:
            results = p.map(self.process_small_batch, df_split)

        for result in results:
            for exam in result:
                for instance in exam:
                    self.append_exam(instance)

    def process_small_batch(self, df):
        curr_exams = []

        with tqdm(
            total=len(df),
            desc=f"Processing cmmd batch {df.index[0]} to {df.index[-1]}",
            unit="exam",
            ncols=100,
            position=0,
            leave=True,
        ) as pbar:
            for _, row in df.iterrows():
                exams = self.process_row(row)
                pbar.update(1)
                curr_exams.append(exams)
        return curr_exams

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

        for i, img_path in enumerate(imgs):
            exam_id = str(uuid.uuid4())
            exam_imgs_path = self.save_imgs(
                img_path, f"{exam_id}-{i}", row["patient id"]
            )

            if isna_v2(exam_imgs_path):
                logging.warning(
                    f"Skipping image for patient {row['patient id']} exam {exam_id} due to processing error."
                )
                continue

            # construct context
            context = JSONReportContext(
                patient_context=PatientContext(age=bin_age(sanitize_age(row["age"]))),
                exam_context=ExamContext(
                    modality=self._modality,
                    laterality=row["laterality"],
                    view=CRANIAL_CAUDAL if i == 0 else MEDIOLATERAL_OBLIQUE,
                ),
            )

            # construct final exam info
            findings = JSONReportFindings(
                breast=BreastFindings(
                    density=get_proper_breast_density(row["breast density"])
                ),
                lesion=(
                    NOT_PRESENT
                    if row["birads"] == birads_mapping[1]
                    else self._construct_primary_lesion(row)
                ),
                assessment=Assessment(birads=get_proper_birads(row["birads"])),
            )

            current_exam = ExamInformation(
                id=exam_id,
                patient=f"{self.get_dataset_name()}-{row['patient id']}",
                dataset=self.get_dataset_name(),
                modality=self._modality,
                birads=get_proper_birads(row["birads"]),
                race=self._race,
                machine=self._machine,
                exam=exam_imgs_path,
                segmentation=None,
                context=context.get_string(),
                findings=findings.get_string(),
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

        return {
            "score": score,
            "is_primary": is_primary,
            "descriptor_count": descriptor_count,
            "type_priority": 2,
            "lesion": Lesion(
                type=MASS,
                location=get_proper_location(location),
                mass_details=MassAbnormality(
                    shape=get_proper_mass_shape(shape),
                    margin=get_proper_mass_margin(margin),
                    density=get_proper_mass_density(density),
                ),
            ),
        }

    def _build_calcification_candidate(
        self, row, prefix: str, is_primary: bool, associated_col: Optional[str] = None
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
                type=CALCIFICATION,
                location=get_proper_location(location),
                calcification_details=CalcificationAbnormality(
                    type=calc_type,
                    type_details=calc_det,
                    distribution=get_proper_calcification_distribution(distribution),
                ),
            ),
        }

    def _construct_primary_lesion(self, row) -> Optional[Lesion]:
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

        best_candidate = max(
            valid_candidates,
            key=lambda c: (
                c["score"],
                c["descriptor_count"],
                c["is_primary"],
                c["type_priority"],
            ),
        )
        return best_candidate["lesion"]

    def save_imgs(self, image_path: str, exam_id: str, patient_id: str):
        res = self.image_processor.process_image(img_path=image_path)
        if res is None:
            return None

        save_path = os.path.join(
            self.image_processor.processed_imgs_path,
            f"{patient_id}-{exam_id}{self.image_processor.IMGS_SUFFIX}",
        )
        return self.image_processor.save_process(save_path, [res])
