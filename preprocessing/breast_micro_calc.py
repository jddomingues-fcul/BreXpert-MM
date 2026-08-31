import logging
import os
import uuid

import pandas as pd

from dtos.breast_cancer_dataset import (
    ASSESSMENT_EXAM_LEVEL,
    BreastCancerDataset,
    ExamInformation,
)
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext, PatientContext
from dtos.json_report_findings import (
    SCOPE_INDEX_ONLY,
    Assessment,
    BreastFindings,
    JSONReportFindings,
    Lesion,
)
from utils.error_handling import trycatch_func
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    BIRADS_MAPPING,
    CALCIFICATION,
    CRANIAL_CAUDAL,
    MAMMOGRAPHY,
    MEDIOLATERAL_OBLIQUE,
    NOT_PRESENT,
    UNKNOWN,
    bin_age,
    csv_column_cleaning,
    get_proper_birads,
    get_proper_breast_density,
    isna_v2,
    map_birads,
    normalize_laterality,
)

raw_imgs_path = "../data/raw/breast-micro-calc/imgs"
raw_imgs_extension = ".dcm"
processed_imgs_path = "../data/processed/breast-micro-calc/imgs"
description_path = "../data/raw/breast-micro-calc/Description.xlsx"
csv_save_path = "../data/processed/breast-micro-calc/breast-micro-calc.csv"


class BreastMicroCalc(BreastCancerDataset):
    _modality = MAMMOGRAPHY
    _machine = UNKNOWN
    _race = UNKNOWN

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(modality=MAMMOGRAPHY),
        )

        self.normal_cases = pd.read_excel(
            description_path, sheet_name="Normal_cases_modified"
        )
        self.suspicious_cases = pd.read_excel(
            description_path, sheet_name="Suspicious_cases_modified"
        )

    def process_info(self):
        # 1. Adjust normal cases
        self.normal_cases.columns = csv_column_cleaning(list(self.normal_cases.columns))
        self.normal_cases = self.normal_cases.rename(
            columns={
                "bi-rads categories for breast density": "breast density",
                "bi-rads categories for classification": "birads",
                "breast right/left": "laterality",
                "age at the time of the recent mammogram": "age",
            }
        )
        self.normal_cases["subfolder"] = "Normal_cases"

        # 2. Adjust suspicious cases
        self.suspicious_cases.columns = csv_column_cleaning(
            list(self.suspicious_cases.columns)
        )
        self.suspicious_cases = self.suspicious_cases.rename(
            columns={
                "bi-rads categories for breast density": "breast density",
                "bi-rads categories for classification": "birads",
                "breast right/left": "laterality",
                "age at the time of the recent mammogram": "age",
            }
        )
        self.suspicious_cases["subfolder"] = "Suspicious_cases"

        # 3. Combine normal and suspicious cases
        breast_micro_calc = pd.concat(
            [self.normal_cases, self.suspicious_cases], axis=0, ignore_index=True
        )
        breast_micro_calc = breast_micro_calc.replace("NOT AVAILABLE", None)

        # 4. Replace the values accordingly
        breast_micro_calc["birads"] = breast_micro_calc["birads"].apply(
            lambda x: map_birads(x, source="Breast-Micro-Calc", modality=MAMMOGRAPHY)
        )
        breast_micro_calc["laterality"] = breast_micro_calc["laterality"].apply(
            normalize_laterality
        )

        self.run_row_pool(breast_micro_calc)

    def get_lesion(self, row):
        if row["subfolder"] == "Suspicious_cases":
            return Lesion(finding_type=CALCIFICATION)

        if row["birads"] == BIRADS_MAPPING[1]:
            return NOT_PRESENT

        return None

    @trycatch_func
    def process_row(self, row):
        cc_recent_id, cc_recent_img = str(uuid.uuid4()), os.path.join(
            raw_imgs_path,
            row["subfolder"],
            str(row["folder #"]),
            f"CC_recent{raw_imgs_extension}",
        )
        mlo_recent_id, mlo_recent_img = str(uuid.uuid4()), os.path.join(
            raw_imgs_path,
            row["subfolder"],
            str(row["folder #"]),
            f"MLO_recent{raw_imgs_extension}",
        )

        patient = f"{row['subfolder']}_{row['folder #']!s}"

        curr_exams = []
        for exam_id, exam_img in [
            (cc_recent_id, cc_recent_img),
            (mlo_recent_id, mlo_recent_img),
        ]:
            # Apparanetly the images are not always in lower case. Random images with the extension in upper case for some reason
            if not os.path.exists(exam_img):
                exam_img = exam_img.replace(
                    raw_imgs_extension,
                    raw_imgs_extension.upper(),
                )

            exam_imgs_path = self.image_processor.process_and_save_image(
                exam_img, exam_id, patient
            )

            if isna_v2(exam_imgs_path):
                logging.warning(
                    f"Image processing failed for patient {patient}, exam {exam_img}. Skipping exam."
                )
                continue

            # construct context
            context = JSONReportContext(
                patient_context=PatientContext(age=bin_age(row["age"])),
                exam_context=ExamContext(
                    modality=self._modality,
                    laterality=row["laterality"],
                    view=CRANIAL_CAUDAL if "CC" in exam_img else MEDIOLATERAL_OBLIQUE,
                ),
            )

            # construct final exam info
            assessment_birads = get_proper_birads(row["birads"], modality=MAMMOGRAPHY)

            findings = JSONReportFindings.build(
                breast=BreastFindings(
                    composition=get_proper_breast_density(row["breast density"])
                ),
                lesion=self.get_lesion(row),
                assessment=Assessment(birads=assessment_birads),
                annotation_scope=SCOPE_INDEX_ONLY,
            )

            # Only the "recent" CC/MLO pair is loaded; the prior exams in each
            # case folder are deliberately left out because the clinical sheet
            # describes the recent study.
            exam = ExamInformation(
                id=exam_id,
                patient=f"{self.get_dataset_name()}-{patient}",
                dataset=self.get_dataset_name(),
                modality=self._modality,
                birads=assessment_birads,
                race=self._race,
                machine=self._machine,
                exam=exam_imgs_path,
                segmentation=None,
                context=context.get_string(),
                findings=findings.get_string(),
                exam_id=f"{self.get_dataset_name()}-{patient}",
                breast_id=(f"{self.get_dataset_name()}-{patient}-{row['laterality']}"),
                assessment_scope=ASSESSMENT_EXAM_LEVEL,
            )

            curr_exams.append(exam)

        return curr_exams
