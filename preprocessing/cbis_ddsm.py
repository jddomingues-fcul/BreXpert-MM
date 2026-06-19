import glob
import logging
import os
import uuid
from functools import partial
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
from dtos.json_report_context import ExamContext, JSONReportContext
from dtos.json_report_findings import (
    Assessment,
    BreastFindings,
    CalcificationAbnormality,
    JSONReportFindings,
    Lesion,
    MassAbnormality,
)
from utils.error_handling import log_func_info, trycatch_func
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    CALCIFICATION,
    MASS,
    NOT_PRESENT,
    UNKNOWN,
    birads_assessment,
    birads_mapping,
    breast_density,
    column_cleaning_csv_reading,
    dview,
    get_proper_birads,
    get_proper_breast_density,
    get_proper_calcification_distribution,
    get_proper_calcification_type,
    get_proper_exam_view,
    get_proper_mass_margin,
    get_proper_mass_shape,
    get_value,
    isna_v2,
    laterality,
    read_breast_image,
)

raw_imgs_path = "../data/raw/cbis-ddsm/jpeg"
raw_imgs_extension = ".jpg"
processed_imgs_path = "../data/processed/cbis-ddsm/imgs"
calc_case_description_train = (
    "../data/raw/cbis-ddsm/calc_case_description_train_set.csv"
)
calc_case_description_test = "../data/raw/cbis-ddsm/calc_case_description_test_set.csv"
mass_case_description_train = (
    "../data/raw/cbis-ddsm/mass_case_description_train_set.csv"
)
mass_case_description_test = "../data/raw/cbis-ddsm/mass_case_description_test_set.csv"
csv_save_path = "../data/processed/cbis-ddsm/cbis-ddsm.csv"


class CbisDDSM(BreastCancerDataset):
    _modality = "mg"
    _machine = ["DBA", "HOWTEK", "LUMYSIS"]
    _race = UNKNOWN

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(
                read_func=partial(read_breast_image)
            ),
        )

        # combine calcifications
        calc_desc_train_df = column_cleaning_csv_reading(calc_case_description_train)
        calc_desc_test_df = column_cleaning_csv_reading(calc_case_description_test)
        calc_desc = pd.concat(
            [calc_desc_train_df, calc_desc_test_df], ignore_index=True
        )

        # combine masses
        mass_desc_train_df = column_cleaning_csv_reading(mass_case_description_train)
        mass_desc_test_df = column_cleaning_csv_reading(mass_case_description_test)
        mass_desc = pd.concat(
            [mass_desc_train_df, mass_desc_test_df], ignore_index=True
        )

        # merge them together
        self.full_df = pd.concat([calc_desc, mass_desc], ignore_index=True)

    def process_info(self):
        self.full_df.drop_duplicates(
            subset=[
                "patient id",
                "breast density",
                "left or right breast",
                "image view",
                "abnormality type",
                "calc type",
                "calc distribution",
                "assessment",
                "pathology",
                "image file path",
                "mass shape",
                "mass margins",
            ],
            inplace=True,
        )

        # 1. if abnormality type is calcification, then mass shape and mass margins should be NOT_APPLICABLE
        self.full_df.loc[
            self.full_df["abnormality type"] == "calcification",
            ["mass shape", "mass margins"],
        ] = None

        # 2. if abnormality type is mass, then calc type and calc distribution should be NOT_APPLICABLE
        self.full_df.loc[
            self.full_df["abnormality type"] == "mass",
            ["calc type", "calc distribution"],
        ] = None

        # 3. Rename columns to be more consistent
        self.full_df = self.full_df.rename(
            columns={
                "left or right breast": "laterality",
                "calc type": "calcification type",
                "calc distribution": "calcification distribution",
            }
        )

        # 4. Adjust column values
        self.full_df["laterality"] = self.full_df["laterality"].apply(
            lambda x: get_value(x, laterality)
        )
        self.full_df["image view"] = self.full_df["image view"].apply(
            lambda x: get_value(x, dview)
        )
        self.full_df["breast density"] = self.full_df["breast density"].apply(
            lambda x: get_value(x, breast_density)
        )
        self.full_df["assessment"] = self.full_df["assessment"].apply(
            lambda x: get_value(x, birads_assessment)
        )

        # 5. Drop additional evaluation birads
        self.full_df = self.full_df[self.full_df["assessment"] != birads_mapping[0]]

        # 6. Second rename cleaning
        self.full_df = self.full_df.rename(
            columns={"image view": "exam view", "mass margins": "mass margin"}
        )

        # Process the exams
        n = cpu_count() - 1
        df_split = np.array_split(self.full_df, n)
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
            desc=f"Processing cbis-ddsm batch {df.index[0]} to {df.index[-1]}",
            unit="exam",
            ncols=100,
            position=0,
            leave=True,
        ) as pbar:
            for _, row in df.iterrows():
                exam = self.process_row(row)
                pbar.update(1)
                if not isna_v2(exam):
                    curr_exams.append(exam)
        return curr_exams

    @trycatch_func
    def process_row(self, row):
        exam_id = str(uuid.uuid4())
        exam_path, img_shape = self.save_slice(
            row["image file path"], exam_id, str(row["patient id"])
        )

        if isna_v2(exam_path):
            logging.warning(
                f"Image processing failed for patient {row['patient id']}, exam {row['image file path']}. Skipping exam."
            )
            return None

        segmentation_path = self.save_segmentation(
            exam_id, str(row["patient id"]), row["roi mask file path"], img_shape
        )

        # construct context
        context = JSONReportContext(
            exam_context=ExamContext(
                modality=self._modality,
                laterality=row["laterality"],
                view=get_proper_exam_view(row["exam view"]),
            )
        )

        tgt_lesion = None
        if row["abnormality type"] == "mass":
            tgt_lesion = Lesion(
                laterality=row["laterality"],
                type=MASS,
                mass_details=MassAbnormality(
                    shape=get_proper_mass_shape(row["mass shape"]),
                    margin=get_proper_mass_margin(row["mass margin"]),
                ),
            )

        elif row["abnormality type"] == "calcification":
            calc_type, calc_details = get_proper_calcification_type(
                row["calcification type"]
            )
            tgt_lesion = Lesion(
                laterality=row["laterality"],
                type=CALCIFICATION,
                calcification_details=CalcificationAbnormality(
                    type=calc_type,
                    type_details=calc_details,
                    distribution=get_proper_calcification_distribution(
                        row["calcification distribution"]
                    ),
                ),
            )

        # construct final exam info
        findings = JSONReportFindings(
            breast=BreastFindings(
                density=get_proper_breast_density(row["breast density"])
            ),
            lesion=(
                NOT_PRESENT if row["assessment"] == birads_mapping[1] else tgt_lesion
            ),
            assessment=Assessment(birads=get_proper_birads(row["assessment"])),
        )

        return ExamInformation(
            id=exam_id,
            patient=f"{self.get_dataset_name()}-{row['patient id']}",
            dataset=self.get_dataset_name(),
            modality=self._modality,
            birads=get_proper_birads(row["assessment"]),
            race=self._race,
            machine=";".join(
                self._machine
            ),  # join them with ; to indicate multiple possible machines
            exam=exam_path,
            segmentation=(
                [segmentation_path] if not isna_v2(segmentation_path) else None
            ),
            context=context.get_string(),
            findings=findings.get_string(),
        )

    @trycatch_func
    @log_func_info
    def save_slice(self, img_file_path, exam_id, patient_id):
        base_folder = img_file_path.split("/")[-2]
        path = os.path.join(
            raw_imgs_path,
            base_folder,
            f"*{raw_imgs_extension}",
        )
        im_path = glob.glob(path, recursive=True)[0]

        img = self.image_processor.process_image(im_path)
        if img is None:
            return None, None

        save_path = os.path.join(
            processed_imgs_path,
            f"{patient_id}-{exam_id}{ImageProcessor.IMGS_SUFFIX}",
        )
        return self.image_processor.save_process(save_path, [img]), img.shape

    @trycatch_func
    @log_func_info
    def save_segmentation(
        self, exam_id, patient_id, roi_image_path, img_shape
    ) -> Optional[str]:
        base_folder = roi_image_path.split("/")[-2]
        path = os.path.join(
            raw_imgs_path,
            base_folder,
            f"*{raw_imgs_extension}",
        )
        available_img_paths = glob.glob(path, recursive=True)

        if not available_img_paths or img_shape is None:
            return None

        mask_path = self.get_adequate_mask_path(available_img_paths)
        seg_mask = self.image_processor.process_segmentation_mask(mask_path)

        if seg_mask is not None:
            seg_save_path = os.path.join(
                processed_imgs_path,
                f"{patient_id}-{exam_id}{ImageProcessor.SEGMENTATION_SUFFIX}",
            )
            return self.image_processor.save_process(seg_save_path, [seg_mask])

        return None

    def get_adequate_mask_path(self, available_img_paths: list):
        if len(available_img_paths) == 1:
            logging.warning(
                f"Only one image found for segmentation mask. Using {available_img_paths[0]}"
            )
            return available_img_paths[0]
        else:
            # read each image and select the one with the highest resolution
            shapes = []
            for img in available_img_paths:
                current_img = self.image_processor.read_image(img)
                if current_img is None:
                    continue
                shapes.append(current_img.shape[0] * current_img.shape[1])

            if len(shapes) == 0:
                return None

            return available_img_paths[np.argmax(shapes)]
