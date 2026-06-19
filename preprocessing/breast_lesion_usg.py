import logging
import os
import uuid
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from dtos.breast_cancer_dataset import BreastCancerDataset, ExamInformation
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext, PatientContext
from dtos.json_report_findings import (
    Assessment,
    BreastFindings,
    JSONReportFindings,
    Lesion,
    MassAbnormality,
)
from utils.error_handling import trycatch_func
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    MASS,
    NOT_PRESENT,
    UNKNOWN,
    bin_age,
    birads_assessment,
    birads_mapping,
    csv_column_cleaning,
    get_proper_birads,
    get_proper_halo,
    get_proper_mass_echotexture,
    get_proper_mass_margin,
    get_proper_mass_shape,
    get_proper_posterior_features,
    get_proper_tissue_composition,
    get_value,
    isna_v2,
    read_breast_image,
    sanitize_age,
)

raw_imgs_path = "../data/raw/breast-lesions-usg/BrEaST-Lesions_USG-images_and_masks"
processed_imgs_path = "../data/processed/breast-lesions-usg/imgs"
lesions_usg_path = (
    "../data/raw/breast-lesions-usg/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
)
csv_save_path = "../data/processed/breast-lesions-usg/breast-lesions-usg.csv"


class BreastLesionUSG(BreastCancerDataset):
    _modality = "us"
    _machines = [
        "hitachi arietta 70 equipped with linear array transducer l441 (frequency range: 2-12 mhz)",
        "esaote 6150 equipped with linear array transducer la523 (frequency range: 4-13 mhz)",
        "samsung rs85 equipped with linear array transducer l3-12a (frequency range: 3-12 mhz)",
        "philips affiniti 70 g and epiq 5 g equipped with linear array transducers el18-4 (frequency range: 2-22 mhz) and l12-5 (frequency range: 5-12 mhz)",
    ]
    _race = UNKNOWN

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.lesions_usg_df = pd.read_excel(lesions_usg_path, sheet_name=0)
        self.lesions_usg_df.columns = csv_column_cleaning(
            list(self.lesions_usg_df.columns)
        )
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(
                read_func=partial(read_breast_image)
            ),
        )

    def process_info(self):
        # 1. Drop columns that are not needed, or that we cannot use because are post biopsy
        self.lesions_usg_df = self.lesions_usg_df.drop(
            columns=["mask other filename", "pixel size", "verification"]
        )

        # 2. Adjust the birads values
        self.lesions_usg_df["birads"] = self.lesions_usg_df["birads"].apply(
            lambda x: get_value(x, birads_assessment)
        )

        # 3. replace not applicable with None, and not available with None
        self.lesions_usg_df = self.lesions_usg_df.replace("not applicable", None)
        self.lesions_usg_df = self.lesions_usg_df.replace("not available", None)

        # Process the exams
        n = cpu_count() - 1
        df_split = np.array_split(self.lesions_usg_df, n)
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
            desc=f"Processing breast lesions usg {df.index[0]} to {df.index[-1]}",
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
        patient_id = f"{self.get_dataset_name()}-{row['case id']}"

        exam_imgs_path = self.image_processor.process_and_save_image(
            os.path.join(raw_imgs_path, row["image filename"]),
            exam_id,
            patient_id,
        )

        if isna_v2(exam_imgs_path):
            logging.warning(
                f"No images were able to be saved for patient {patient_id}. Skipping this exam."
            )
            return None

        if isna_v2(row["mask tumor filename"]):
            segmentation_path = None
        else:
            segmentation_path = self.image_processor.save_segmentation_mask(
                os.path.join(raw_imgs_path, row["mask tumor filename"]),
                exam_id,
                patient_id,
            )

        context = JSONReportContext(
            patient_context=PatientContext(age=bin_age(sanitize_age(row["age"]))),
            exam_context=ExamContext(modality=self._modality),
        )

        # construct final exam info
        findings = JSONReportFindings(
            breast=BreastFindings(
                tissue_composition=get_proper_tissue_composition(
                    row["tissue composition"]
                )
            ),
            lesion=(
                NOT_PRESENT
                if row["birads"] == birads_mapping[1]
                else self._construct_lesions_array(row)
            ),
            assessment=Assessment(birads=get_proper_birads(row["birads"])),
        )

        return ExamInformation(
            id=exam_id,
            patient=patient_id,
            dataset=self.get_dataset_name(),
            modality=self._modality,
            birads=get_proper_birads(row["birads"]),
            race=self._race,
            machine=";".join(self._machines),
            exam=exam_imgs_path,
            segmentation=(
                [segmentation_path] if not isna_v2(segmentation_path) else None
            ),
            context=context.get_string(),
            findings=findings.get_string(),
        )

    def _construct_lesions_array(self, row) -> Lesion:
        return Lesion(
            type=MASS,
            mass_details=MassAbnormality(
                shape=get_proper_mass_shape(row["shape"]),
                margin=get_proper_mass_margin(row["margin"]),
                echogenicity=get_proper_mass_echotexture(row["echogenicity"]),
                posterior_features=get_proper_posterior_features(
                    row["posterior features"]
                ),
                halo=get_proper_halo(row["halo"]),
            ),
        )
