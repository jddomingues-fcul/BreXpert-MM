import glob
import logging
import os
import uuid
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from dtos.breast_cancer_dataset import (
    ASSESSMENT_GROUNDED,
    BreastCancerDataset,
    ExamInformation,
)
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext
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
from utils.error_handling import log_func_info, pool_worker, trycatch_func
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    BIRADS_MAPPING,
    CALCIFICATION,
    MAMMOGRAPHY,
    MASS,
    UNKNOWN,
    birads_category_number,
    column_cleaning_csv_reading,
    dview,
    get_proper_birads,
    get_proper_breast_density,
    get_proper_calcification_distribution,
    get_proper_calcification_type,
    get_proper_exam_view,
    get_proper_finding_type,
    get_proper_mass_shape,
    get_value,
    isna_v2,
    map_birads,
    migrate_mammography_microlobulated,
    normalize_laterality,
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
    _modality = MAMMOGRAPHY
    _machine = UNKNOWN
    _race = UNKNOWN

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(
                modality=MAMMOGRAPHY,
                read_func=partial(read_breast_image),
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
            normalize_laterality
        )
        self.full_df["image view"] = self.full_df["image view"].apply(
            lambda x: get_value(x, dview)
        )
        # "breast density" stays as the raw ACR code; get_proper_breast_density
        # maps 1-4 / a-d straight onto the v2025 composition statements.
        self.full_df["assessment"] = self.full_df["assessment"].apply(
            lambda x: map_birads(x, source="CBIS-DDSM", modality=MAMMOGRAPHY)
        )

        # 5. Drop additional evaluation birads
        self.full_df = self.full_df[self.full_df["assessment"] != BIRADS_MAPPING[0]]

        # 6. Second rename cleaning
        self.full_df = self.full_df.rename(
            columns={"image view": "exam view", "mass margins": "mass margin"}
        )

        # 7. Rank the abnormalities within each image
        self.full_df = self.rank_abnormalities(self.full_df)

        # Process the exams, splitting on image boundaries so that every row of
        # one image lands in the same worker.
        self.report.rows_seen = len(self.full_df)

        image_ids = self.full_df["image file path"].unique()
        n = max(1, cpu_count() - 1)
        batches = [
            self.full_df[self.full_df["image file path"].isin(chunk)]
            for chunk in np.array_split(image_ids, n)
            if len(chunk) > 0
        ]

        with Pool(processes=n) as p:
            results = p.map(self.process_small_batch, batches)

        self.collect_pool_results(results)

    def rank_abnormalities(self, df: pd.DataFrame) -> pd.DataFrame:
        ranked = df.copy()

        ranked["_birads_rank"] = (
            ranked["assessment"]
            .map(birads_category_number, na_action="ignore")
            .fillna(-1)
            .astype(int)
        )
        descriptor_cols = [
            "mass shape",
            "mass margin",
            "calcification type",
            "calcification distribution",
        ]
        ranked["_descriptor_rank"] = (
            ranked[descriptor_cols].notna().sum(axis=1).astype(int)
        )
        # Masses are the more specific finding when an image carries both.
        ranked["_type_rank"] = (ranked["abnormality type"] == "mass").astype(int)

        ranked = ranked.sort_values(
            by=[
                "image file path",
                "_birads_rank",
                "_descriptor_rank",
                "_type_rank",
            ],
            ascending=[True, False, False, False],
            kind="mergesort",
        ).reset_index(drop=True)

        per_image = ranked.groupby("image file path").size()
        logging.info(
            f"CBIS-DDSM: {len(ranked)} abnormality rows across {len(per_image)} images "
            f"({int((per_image > 1).sum())} images carry more than one finding, "
            f"max {int(per_image.max())})."
        )
        return ranked

    @pool_worker
    def process_small_batch(self, df):
        curr_exams = []
        groups = list(df.groupby("image file path", sort=False))

        with tqdm(
                total=len(groups),
                desc="Processing cbis-ddsm batch",
                unit="image",
                ncols=100,
                position=0,
                leave=True,
        ) as pbar:
            for _, group in groups:
                exam = self.process_image(group)
                pbar.update(1)
                if exam is not None:
                    curr_exams.append(exam)
        return curr_exams

    @trycatch_func
    def process_image(self, group: pd.DataFrame):
        lead = group.iloc[0]
        exam_id = str(uuid.uuid4())
        patient_id = str(lead["patient id"])

        exam_path, img_shape = self.save_slice(
            lead["image file path"], exam_id, patient_id
        )

        if isna_v2(exam_path):
            logging.warning(
                f"Image processing failed for patient {patient_id}, "
                f"exam {lead['image file path']}. Skipping exam."
            )
            return None

        segmentation_paths = []
        for position, (_, row) in enumerate(group.iterrows()):
            path = self.save_segmentation(
                f"{exam_id}-L{position + 1}",
                patient_id,
                row["roi mask file path"],
                img_shape,
            )
            if not isna_v2(path):
                segmentation_paths.append(path)

        context = JSONReportContext(
            exam_context=ExamContext(
                modality=self._modality,
                laterality=lead["laterality"],
                view=get_proper_exam_view(lead["exam view"]),
            )
        )

        lesions = [self.construct_lesion(row) for _, row in group.iterrows()]
        lesions = [lesion for lesion in lesions if lesion is not None]
        if lesions:
            lesions[0].is_index = True

        assessment_birads = get_proper_birads(lead["assessment"], modality=MAMMOGRAPHY)

        findings = JSONReportFindings.build(
            breast=BreastFindings(
                composition=get_proper_breast_density(lead["breast density"])
            ),
            lesion=lesions if lesions else None,
            assessment=Assessment(birads=assessment_birads),
            annotation_scope=SCOPE_COMPLETE,
        )

        return ExamInformation(
            id=exam_id,
            patient=f"{self.get_dataset_name()}-{patient_id}",
            dataset=self.get_dataset_name(),
            modality=self._modality,
            birads=assessment_birads,
            race=self._race,
            machine=self._machine,
            exam=exam_path,
            segmentation=segmentation_paths or None,
            context=context.get_string(),
            findings=findings.get_string(),
            exam_id=f"{self.get_dataset_name()}-{patient_id}-{lead['laterality']}",
            breast_id=f"{self.get_dataset_name()}-{patient_id}-{lead['laterality']}",
            assessment_scope=ASSESSMENT_GROUNDED,
        )

    def construct_lesion(self, row) -> Lesion | None:
        location = LesionLocation(laterality=row["laterality"])

        if row["abnormality type"] == "mass":
            shape_value = row["mass shape"]
            routed = (
                get_proper_finding_type(shape_value, MAMMOGRAPHY)
                if not isna_v2(shape_value)
                   and get_proper_mass_shape(shape_value) is None
                else None
            )

            if routed is not None and routed != MASS:
                return Lesion(location=location, finding_type=routed)

            shape, margin = migrate_mammography_microlobulated(
                shape_value, row["mass margin"]
            )
            return Lesion(
                location=location,
                finding_type=MASS,
                mass_details=MassAbnormality(shape=shape, margin=margin),
            )

        if row["abnormality type"] == "calcification":
            calc_type, calc_details = get_proper_calcification_type(
                row["calcification type"]
            )
            return Lesion(
                location=location,
                finding_type=CALCIFICATION,
                calcification_details=CalcificationAbnormality(
                    type=calc_type,
                    type_details=calc_details,
                    distribution=get_proper_calcification_distribution(
                        row["calcification distribution"]
                    ),
                ),
            )

        return None

    @trycatch_func
    @log_func_info
    def save_slice(self, img_file_path, exam_id, patient_id):
        base_folder = img_file_path.split("/")[-2]
        path = os.path.join(
            raw_imgs_path,
            base_folder,
            f"*{raw_imgs_extension}",
        )
        matches = glob.glob(path, recursive=True)
        if not matches:
            logging.warning(f"No full image found under {path}.")
            return None, None

        im_path = matches[0]
        raw_img = self.image_processor.read_image(im_path)
        if raw_img is None:
            return None, None

        # Return the *raw* shape: the caller uses it to pick the ROI file that
        # shares the image's coordinate frame, and every processed image is
        # 512x512 so the processed shape carries no information.
        raw_shape = raw_img.shape

        img = self.image_processor.apply_processing(raw_img, is_segmentation=False)
        if img is None:
            return None, None

        save_path = os.path.join(
            processed_imgs_path,
            f"{patient_id}-{exam_id}{ImageProcessor.IMGS_SUFFIX}",
        )
        return self.image_processor.save_process(save_path, [img]), raw_shape

    @trycatch_func
    @log_func_info
    def save_segmentation(
            self, exam_id, patient_id, roi_image_path, img_shape
    ) -> str | None:
        base_folder = roi_image_path.split("/")[-2]
        path = os.path.join(
            raw_imgs_path,
            base_folder,
            f"*{raw_imgs_extension}",
        )
        available_img_paths = sorted(glob.glob(path, recursive=True))

        if not available_img_paths or img_shape is None:
            return None

        mask_path = self.get_adequate_mask_path(available_img_paths, img_shape)
        if mask_path is None:
            logging.warning(
                f"No ROI file matching the full image geometry {img_shape} under {path}."
            )
            return None

        seg_mask = self.image_processor.process_segmentation_mask(mask_path)

        if seg_mask is not None:
            seg_save_path = os.path.join(
                processed_imgs_path,
                f"{patient_id}-{exam_id}{ImageProcessor.SEGMENTATION_SUFFIX}",
            )
            return self.image_processor.save_process(seg_save_path, [seg_mask])

        return None

    def get_adequate_mask_path(self, available_img_paths: list, img_shape):
        for candidate in available_img_paths:
            current_img = self.image_processor.read_image(candidate)
            if current_img is None:
                continue
            if current_img.shape == img_shape:
                return candidate

        return None
