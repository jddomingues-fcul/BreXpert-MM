import logging
import os
import uuid
from multiprocessing import Pool, cpu_count

import numpy as np
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
    column_cleaning_csv_reading,
    dview,
    get_proper_birads,
    get_proper_breast_density,
    get_proper_exam_view,
    get_value,
    isna_v2,
    laterality,
    sanitize_age,
    yes_no_mapping,
)

raw_imgs_path = "../data/raw/rsna-bcd"
raw_imgs_extension = ".dcm"
processed_imgs_path = "../data/processed/rsna-bcd/imgs"
train_csv_path = "../data/raw/rsna-bcd/train.csv"
csv_save_path = "../data/processed/rsna-bcd/rsna-bcd.csv"


class RsnaBCD(BreastCancerDataset):
    _modality = "mg"
    _machine = UNKNOWN
    _race = UNKNOWN

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.train_df = column_cleaning_csv_reading(train_csv_path)

        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(),
        )

    def process_info(self):
        # Drop irrelevant columns
        train_cols_to_drop = ["site id", "machine id", "difficult negative case"]
        self.train_df = self.train_df.drop(columns=train_cols_to_drop)

        # Filter for BI-RADS not nan
        self.train_df = self.train_df[self.train_df["birads"].notna()]

        # Filter to exclude BI-RADS 0
        self.train_df = self.train_df[self.train_df["birads"] != 0]

        # Add column for set of images folder
        self.train_df["raw folder"] = "train_images"

        # Birads mapping, cancer, biopsy, implant, and invasive mapping to yes and no responses
        self.train_df["birads"] = self.train_df["birads"].apply(
            lambda x: get_value(int(x), birads_assessment)
        )
        self.train_df["cancer"] = self.train_df["cancer"].apply(
            lambda x: get_value(x, yes_no_mapping)
        )
        self.train_df["biopsy"] = self.train_df["biopsy"].apply(
            lambda x: get_value(x, yes_no_mapping)
        )
        self.train_df["implant"] = self.train_df["implant"].apply(
            lambda x: get_value(x, yes_no_mapping)
        )
        self.train_df["invasive"] = self.train_df["invasive"].apply(
            lambda x: get_value(x, yes_no_mapping)
        )
        self.train_df["laterality"] = self.train_df["laterality"].apply(
            lambda x: get_value(x, laterality)
        )
        self.train_df["density"] = self.train_df["density"].apply(
            lambda x: get_value(x, breast_density) if not isna_v2(x) else None
        )
        self.train_df["view"] = self.train_df["view"].apply(
            lambda x: get_value(x, dview)
        )

        self.train_df = self.train_df[
            self.train_df["view"].isin([MEDIOLATERAL_OBLIQUE, CRANIAL_CAUDAL])
        ].reset_index(
            drop=True
        )  # filter for only the two most common views, to avoid issues with very small number of samples in other views

        # merge both dfs and rename cols
        col_rename = {
            "implant": "breast implants",
            "density": "breast density",
            "view": "exam view",
        }
        self.train_df = self.train_df.rename(columns=col_rename)

        # Process the exams
        n = cpu_count() - 1
        df_split = [
            self.train_df.iloc[idx]
            for idx in np.array_split(np.arange(len(self.train_df)), n)
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

        # Im sorry
        with tqdm(
            total=len(df),
            desc=f"Processing advanced rsna bcd batch {df.index[0]} to {df.index[-1]}",
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
        # get needed info for exam
        exam_id = str(uuid.uuid4())
        img_path = os.path.join(
            raw_imgs_path,
            row["raw folder"],
            str(row["patient id"]),
            f"{str(row['image id'])}{raw_imgs_extension}",
        )
        exam_imgs_path = self.image_processor.process_and_save_image(
            img_path, exam_id, str(row["patient id"])
        )
        if isna_v2(exam_imgs_path):
            logging.warning(
                f"Skipping exam for patient {row['patient id']} due to processing issues."
            )
            return None

        # construct context
        context = JSONReportContext(
            patient_context=PatientContext(
                age=bin_age(sanitize_age(row["age"])),
                has_implants=row["breast implants"],
            ),
            exam_context=ExamContext(
                modality=self._modality,
                laterality=row["laterality"],
                view=get_proper_exam_view(row["exam view"]),
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

        return ExamInformation(
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
