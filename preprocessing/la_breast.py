import logging
import os
import uuid
from functools import partial
from glob import glob

import numpy as np
import pandas as pd

from dtos.breast_cancer_dataset import (
    ASSESSMENT_GROUNDED,
    BreastCancerDataset,
    ExamInformation,
)
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext
from dtos.json_report_findings import (
    SCOPE_UNKNOWN,
    Assessment,
    BreastFindings,
    JSONReportFindings,
)
from utils.error_handling import pool_worker
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    BILATERAL,
    BIRADS_MAPPING,
    MRI,
    NOT_PRESENT,
    PRESENT,
    UNKNOWN,
    collapse_known_biopsy_proven,
    column_cleaning_csv_reading,
    draw_rectangle_on_image,
    get_proper_birads,
    get_proper_contrast_phase,
    get_proper_mri_fgt,
    isna_v2,
    map_birads,
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
    _modality = MRI
    _race = UNKNOWN
    _machine = "Multiple 1.5T scanners"
    SEQUENCE_COLUMNS = ["d0", "d1", "d2", "d3", "d4", "d5"]

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(
                modality=MRI, read_func=partial(read_breast_image)
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
            lambda x: map_birads(x, source="LA-Breast", modality=MRI)
        )

        # 2. Move birads 6 values to birads 5
        self.clinical_df["birads"] = self.clinical_df["birads"].apply(
            collapse_known_biopsy_proven
        )

        # 3. Drop rows we cannot assess. process_patient calls get_proper_birads,
        # which raises on an unmappable value and would take the whole pool down.
        unmappable = int(self.clinical_df["birads"].isna().sum())
        if unmappable:
            logging.warning(
                f"LA-Breast: dropping {unmappable} rows with an unmappable BI-RADS."
            )
        self.clinical_df = self.clinical_df[
            self.clinical_df["birads"].notna()
        ].reset_index(drop=True)

        # column renameing
        self.clinical_df.rename(columns={"acr": "breast density"}, inplace=True)

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

        self.run_pool(
            self.process_patient, grouped_patients, "Processing LA Breast patients"
        )

    @pool_worker
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

                        processed_img, img_shape = self.process_slice(im_path)
                        if processed_img is None:
                            logging.warning(
                                f"Skipping exam for patient {patient}: could not process {im_path}."
                            )
                            continue

                        segmentation_path = self.create_and_save_segmentation(
                            exam_id, patient_id, img_shape, (dx, dy, cx, cy)
                        )
                        if isna_v2(segmentation_path):
                            logging.warning(
                                f"Skipping exam for patient {patient}: no usable ROI for {im_path}."
                            )
                            continue

                        exam_imgs_path = self.image_processor.save_image_set(
                            [processed_img], exam_id, patient_id
                        )
                        if isna_v2(exam_imgs_path):
                            logging.warning(
                                f"Skipping exam for patient {patient}: could not save {im_path}."
                            )
                            continue

                        context = JSONReportContext(
                            exam_context=ExamContext(
                                modality=self._modality,
                                laterality=BILATERAL,
                                contrast_phase=get_proper_contrast_phase(im_type),
                            ),
                        )

                        assessment_birads = get_proper_birads(
                            row["birads"], modality=MRI
                        )

                        findings = JSONReportFindings.build(
                            breast=BreastFindings(
                                fgt=get_proper_mri_fgt(row["breast density"])
                            ),
                            lesion=(
                                NOT_PRESENT
                                if row["birads"] == BIRADS_MAPPING[1]
                                else PRESENT
                            ),
                            assessment=Assessment(birads=assessment_birads),
                            annotation_scope=SCOPE_UNKNOWN,
                        )

                        exam = ExamInformation(
                            id=exam_id,
                            patient=patient_id,
                            dataset=self.get_dataset_name(),
                            modality=self._modality,
                            birads=assessment_birads,
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
                            exam_id=patient_id,
                            breast_id=None,  # bilateral acquisition
                            assessment_scope=ASSESSMENT_GROUNDED,
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

    def process_slice(self, file_path):
        img = self.image_processor.read_image(file_path)

        if isna_v2(img):
            logging.warning(f"Could not read image for {file_path}")
            return None, None

        img_shape = img.shape
        img = self.image_processor.apply_processing(img, is_segmentation=False)

        if isna_v2(img):
            logging.warning(f"Could not process image for {file_path}")
            return None, None

        return img, img_shape

    def create_and_save_segmentation(self, exam_id, patient_id, img_shape, coordinates):
        dx, dy, cx, cy = coordinates

        if any(isna_v2(v) for v in (dx, dy, cx, cy)):
            logging.warning(
                f"Missing ROI coordinates for patient {patient_id}, exam {exam_id}."
            )
            return None

        height, width = img_shape
        # cv2 takes (x=column, y=row); `centro x`/`distancia x` are column-space.
        x1, y1 = int(cx - dx / 2), int(cy - dy / 2)
        x2, y2 = int(cx + dx / 2), int(cy + dy / 2)

        # A box that lands entirely outside the frame would otherwise be written
        # out as a valid-looking, wholly empty annotation.
        if x2 <= 0 or y2 <= 0 or x1 >= width or y1 >= height:
            logging.warning(
                f"ROI box ({x1},{y1})-({x2},{y2}) falls outside the {img_shape} frame "
                f"for patient {patient_id}, exam {exam_id}."
            )
            return None

        # draw_rectangle_on_image returns a copy; ignoring it leaves an empty mask.
        mask = draw_rectangle_on_image(
            np.zeros(img_shape, dtype=np.uint8), x1, y1, x2, y2
        )
        if mask is None or mask.max() == 0:
            logging.warning(
                f"ROI box ({x1},{y1})-({x2},{y2}) produced an empty mask for patient "
                f"{patient_id}, exam {exam_id}."
            )
            return None

        seg_mask = self.image_processor.apply_processing(mask, is_segmentation=True)
        return self.image_processor.save_segmentation_set(
            [seg_mask], exam_id, patient_id
        )

    def _sample_one_row_per_clinical_group(
            self, df: pd.DataFrame, random_state: int = 42
    ) -> pd.DataFrame:
        # Keep one random representative row per (patient, birads, breast density)
        if len(df) == 0:
            return df.copy()

        # `breast density` is missing on a fifth of the rows. GroupBy.sample cannot
        # index a group whose key contains NaN even with dropna=False, so the key is
        # materialised as text with an explicit placeholder first.
        group_key = (
            df[["patient", "birads", "breast density"]]
            .astype(object)
            .apply(lambda col: col.map(lambda v: "<missing>" if isna_v2(v) else str(v)))
            .agg("\x1f".join, axis=1)
        )

        sampled = (
            df.groupby(group_key, sort=False, group_keys=False)
            .sample(n=1, random_state=random_state)
            .reset_index(drop=True)
        )
        return sampled
