import logging
import os
import uuid
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from dtos.breast_cancer_dataset import (
    BreastCancerDataset,
    ExamInformation,
)
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext, PatientContext
from dtos.json_report_findings import Assessment, BreastFindings, JSONReportFindings
from utils.error_handling import trycatch_func
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    CRANIAL_CAUDAL,
    MEDIOLATERAL_OBLIQUE,
    NOT_PRESENT,
    UNKNOWN,
    bin_age,
    birads_assessment,
    birads_mapping,
    breast_density,
    csv_column_cleaning,
    get_proper_birads,
    get_proper_breast_density,
    get_value,
    isna_v2,
    laterality,
    sanitize_age,
)

raw_imgs_path = "../data/raw/breast-micro-calc/imgs"
raw_imgs_extension = ".dcm"
processed_imgs_path = "../data/processed/breast-micro-calc/imgs"
description_path = "../data/raw/breast-micro-calc/Description.xlsx"
csv_save_path = "../data/processed/breast-micro-calc/breast-micro-calc.csv"


class BreastMicroCalc(BreastCancerDataset):
    _modality = "mg"
    _machine = UNKNOWN
    _race = UNKNOWN

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(),
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
            lambda x: get_value(x, birads_assessment)
        )
        breast_micro_calc["laterality"] = breast_micro_calc["laterality"].apply(
            lambda x: get_value(x, laterality)
        )
        breast_micro_calc["breast density"] = breast_micro_calc["breast density"].apply(
            lambda x: get_value(x, breast_density)
        )

        # Process the exams
        n = cpu_count() - 1
        df_split = [
            breast_micro_calc.iloc[idx]
            for idx in np.array_split(np.arange(len(breast_micro_calc)), n)
            if len(idx) > 0
        ]

        with Pool(processes=n) as p:
            results = p.map(self.process_small_batch, df_split)

        for result in results:
            for exam in result:
                self.append_exam(exam)

    @trycatch_func
    def process_small_batch(self, df):
        curr_exams = []

        with tqdm(
            total=len(df),
            desc=f"Processing breast-micro-calc batch {df.index[0]} to {df.index[-1]}",
            unit="exam",
            ncols=100,
            position=0,
            leave=True,
        ) as pbar:
            for _, row in df.iterrows():
                exams = self.process_row(row)
                curr_exams.extend(exams)
                pbar.update(1)
        return curr_exams

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

        patient = f"{row['subfolder']}_{str(row['folder #'])}"

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
                patient_context=PatientContext(age=bin_age(sanitize_age(row["age"]))),
                exam_context=ExamContext(
                    modality=self._modality,
                    laterality=row["laterality"],
                    view=CRANIAL_CAUDAL if "CC" in exam_img else MEDIOLATERAL_OBLIQUE,
                ),
            )

            # construct final exam info
            findings = JSONReportFindings(
                breast=BreastFindings(
                    density=get_proper_breast_density(row["breast density"])
                ),
                lesion=NOT_PRESENT if row["birads"] == birads_mapping[1] else None,
                assessment=Assessment(birads=get_proper_birads(row["birads"])),
            )

            # We save all exams, but only the most recent will have context, findings, and birads. We do this, to keep previous exams as reference if at some point we can use them as context.
            exam = ExamInformation(
                id=exam_id,
                patient=f"{self.get_dataset_name()}-{patient}",
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

            curr_exams.append(exam)

        return curr_exams
