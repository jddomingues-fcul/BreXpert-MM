import logging
import os
import uuid
from functools import partial
from glob import glob
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from dtos.breast_cancer_dataset import (
    BreastCancerDataset,
    ExamInformation,
)
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext
from dtos.json_report_findings import Assessment, BreastFindings, JSONReportFindings
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    BILATERAL,
    NOT_PRESENT,
    birads_assessment,
    birads_mapping,
    breast_density,
    column_cleaning_csv_reading,
    draw_rectangle_on_image,
    get_proper_birads,
    get_proper_breast_density,
    get_proper_contrast_phase,
    get_value,
    isna_v2,
    read_breast_image,
)

raw_imgs_path: str = "../data/raw/la-breast/imgs"
raw_imgs_extension: str = ".tiff"
processed_imgs_path: str = "../data/processed/la-breast/imgs"
train_csv_path: str = "../data/raw/la-breast/train.csv"
test_csv_path: str = "../data/raw/la-breast/test.csv"
val_csv_path: str = "../data/raw/la-breast/val.csv"
csv_save_path: str = "../data/processed/la-breast/la-breast.csv"


class LABreast(BreastCancerDataset):

    _modality = "mr"
    _race = "latin american"
    _machine = "Multiple 1.5T scanners"
    SEQUENCE_COLUMNS = ["d0", "d1", "d2", "d3", "d4", "d5"]

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(
                read_func=partial(read_breast_image)
            ),
        )

        self.train_df = column_cleaning_csv_reading(train_csv_path)
        self.test_df = column_cleaning_csv_reading(test_csv_path)
        self.val_df = column_cleaning_csv_reading(val_csv_path)
        self.clinical_df = pd.concat(
            [self.train_df, self.test_df, self.val_df], ignore_index=True
        )

    def process_info(self):
        # 1. Adjust the birads values
        self.clinical_df["birads"] = self.clinical_df["birads"].apply(
            lambda x: get_value(x, birads_assessment)
        )

        # 2. Move birads 6 values to birads 5
        self.clinical_df["birads"] = self.clinical_df["birads"].replace(
            birads_mapping[6], birads_mapping[5]
        )

        # 3. Adjust breast density values
        self.clinical_df["acr"] = self.clinical_df["acr"].apply(
            lambda x: get_value(int(x), breast_density) if pd.notna(x) else None
        )

        # column renameing
        self.clinical_df.rename(
            columns={
                "acr": "breast density",
            },
            inplace=True,
        )

        # drop columns that are not useful as images:
        keep_cols = [
            "patient",
            "roi",
            "birads",
            "breast density",
            "distancia x d0",
            "distancia y d0",
            "centro x d0",
            "centro y d0",
            "centro z d0",
            "d0",
            "d1",
            "d2",
            "d3",
            "d4",
            "d5",
            "centro x d1",
            "centro y d1",
            "centro z d1",
            "distancia x d1",
            "distancia y d1",
            "centro x d2",
            "centro y d2",
            "centro z d2",
            "distancia x d2",
            "distancia y d2",
            "centro x d3",
            "centro y d3",
            "centro z d3",
            "distancia x d3",
            "distancia y d3",
            "centro x d4",
            "centro y d4",
            "centro z d4",
            "distancia x d4",
            "distancia y d4",
            "centro x d5",
            "centro y d5",
            "centro z d5",
            "distancia x d5",
            "distancia y d5",
        ]
        self.clinical_df = self.clinical_df[keep_cols]

        # Keep one random row per (patient, birads, breast density)
        dedup_df = self._sample_one_row_per_clinical_group(
            self.clinical_df, random_state=42
        )

        # Split deduplicated DataFrame by patient id
        grouped_patients = [group for _, group in dedup_df.groupby("patient")]

        # Process the exams
        n = cpu_count() - 1
        with Pool(processes=n) as p:
            results = list(
                tqdm(
                    p.imap(self.process_patient, grouped_patients),
                    total=len(grouped_patients),
                    desc="Processing LA Breast patients",
                    unit="patient",
                    ncols=100,
                    position=0,
                    leave=True,
                )
            )
        for result in results:
            for exam in result:
                self.append_exam(exam)

    def process_patient(self, patient_df):
        curr_exams = []

        if len(patient_df) == 0:
            logging.warning("Empty patient DataFrame, skipping...")
            return curr_exams

        patient = patient_df.iloc[0]["patient"].replace("Breast_", "")
        patient_imgs = glob(
            os.path.join(raw_imgs_path, "**", "**", f"{patient}*{raw_imgs_extension}")
        )

        allowed_im_types = [c for c in self.SEQUENCE_COLUMNS if c in patient_df.columns]

        for _, row in patient_df.iterrows():
            roi_value = str(row["roi"]).lower()

            for im_type in allowed_im_types:
                cell_value = row.get(im_type, None)

                if pd.isna(cell_value) or not str(cell_value).strip():
                    continue

                curr_type_value = str(cell_value).split("/")[-1].split(".")[0]
                found_value = False

                for im_path in patient_imgs:
                    if (
                        curr_type_value.lower() in im_path.lower()
                        and roi_value in im_path.lower()
                    ):
                        dx = row.get(f"distancia x {im_type}")
                        dy = row.get(f"distancia y {im_type}")
                        cx = row.get(f"centro x {im_type}")
                        cy = row.get(f"centro y {im_type}")

                        exam_id = str(uuid.uuid4())
                        patient_id = f"{self.get_dataset_name()}-{row['patient']}"

                        exam_imgs_path, img_shape = self.process_and_save_slice(
                            im_path, exam_id, patient_id
                        )
                        segmentation_path = (
                            self.create_and_save_segmentation(
                                exam_id, patient_id, img_shape, (dx, dy, cx, cy)
                            )
                            if not isna_v2(exam_imgs_path)
                            else None
                        )

                        if isna_v2(exam_imgs_path) or isna_v2(segmentation_path):
                            logging.warning(
                                f"Skipping exam for patient {patient} because of processing issues. "
                                f"Exams imgs path: {exam_imgs_path}, Segmentation path: {segmentation_path}"
                            )
                            continue

                        context = JSONReportContext(
                            exam_context=ExamContext(
                                modality=self._modality,
                                laterality=BILATERAL,
                                contrast_phase=get_proper_contrast_phase(im_type),
                            ),
                        )

                        findings = JSONReportFindings(
                            breast=BreastFindings(
                                density=get_proper_breast_density(row["breast density"])
                            ),
                            lesion=(
                                NOT_PRESENT
                                if row["birads"] == birads_mapping[1]
                                else None
                            ),
                            assessment=Assessment(
                                birads=get_proper_birads(row["birads"])
                            ),
                        )

                        exam = ExamInformation(
                            id=exam_id,
                            patient=patient_id,
                            dataset=self.get_dataset_name(),
                            modality=self._modality,
                            birads=get_proper_birads(row["birads"]),
                            race=self._race,
                            machine=self._machine,
                            exam=exam_imgs_path,
                            segmentation=(
                                [segmentation_path]
                                if not isna_v2(segmentation_path)
                                else None
                            ),
                            context=context.get_string(),
                            findings=findings.get_string(),
                        )
                        curr_exams.append(exam)

                        found_value = True
                        break

                if not found_value:
                    logging.warning(
                        f"No image matched for patient={patient}, im_type={im_type}, "
                        f"value={curr_type_value}, roi={roi_value}"
                    )

        return curr_exams

    def process_and_save_slice(self, file_path, exam_id, patient_id):
        img = self.image_processor.read_image(file_path)

        if isna_v2(img):
            logging.warning(f"Could not read image for {file_path}")
            return None, None

        img_shape = img.shape
        img = self.image_processor.apply_processing(img, is_segmentation=False)

        if isna_v2(img):
            logging.warning(f"Could not process image for {file_path}")
            return None, None

        img_save_path = self.image_processor.save_image_set([img], exam_id, patient_id)
        return img_save_path, img_shape

    def create_and_save_segmentation(self, exam_id, patient_id, img_shape, coordinates):
        dx, dy, cx, cy = coordinates

        mask = np.zeros(img_shape, dtype=np.uint16)
        x1, y1 = int(cx - dx / 2), int(cy - dy / 2)
        x2, y2 = int(cx + dx / 2), int(cy + dy / 2)
        draw_rectangle_on_image(mask, x1, y1, x2, y2)

        seg_mask = self.image_processor.apply_processing(mask, is_segmentation=True)
        seg_mask_path = self.image_processor.save_segmentation_set(
            [seg_mask], exam_id, patient_id
        )
        return seg_mask_path

    def _sample_one_row_per_clinical_group(
        self, df: pd.DataFrame, random_state: int = 42
    ) -> pd.DataFrame:
        # Keep one random representative row per (patient, birads, breast density)
        if len(df) == 0:
            return df.copy()

        sampled = (
            df.groupby(
                ["patient", "birads", "breast density"], dropna=False, group_keys=False
            )
            .sample(n=1, random_state=random_state)
            .reset_index(drop=True)
        )
        return sampled
