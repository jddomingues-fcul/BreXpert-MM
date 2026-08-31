import logging
import os
import uuid
from functools import partial

import pandas as pd

from dtos.breast_cancer_dataset import (
    ASSESSMENT_GROUNDED,
    BreastCancerDataset,
    ExamInformation,
)
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext, PatientContext
from dtos.json_report_findings import (
    SCOPE_COMPLETE,
    Assessment,
    BreastFindings,
    JSONReportFindings,
    Lesion,
    MassAbnormality,
)
from utils.error_handling import trycatch_func
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    BIRADS_MAPPING,
    MASS,
    NOT_PRESENT,
    ULTRASOUND,
    UNKNOWN,
    bin_age,
    collapse_known_biopsy_proven,
    csv_column_cleaning,
    get_proper_birads,
    get_proper_echogenic_feature,
    get_proper_mass_echo_pattern,
    get_proper_mass_margin,
    get_proper_mass_shape,
    get_proper_posterior_features,
    get_proper_tissue_composition,
    isna_v2,
    map_birads,
    read_breast_image,
)

raw_imgs_path = "../data/raw/breast-lesions-usg/BrEaST-Lesions_USG-images_and_masks"
processed_imgs_path = "../data/processed/breast-lesions-usg/imgs"
lesions_usg_path = (
    "../data/raw/breast-lesions-usg/BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
)
csv_save_path = "../data/processed/breast-lesions-usg/breast-lesions-usg.csv"


class BreastLesionUSG(BreastCancerDataset):
    _machine = UNKNOWN
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
                modality=ULTRASOUND, read_func=partial(read_breast_image)
            ),
        )

    def process_info(self):
        # 1. Drop columns that are not needed, or that we cannot use because are post biopsy
        self.lesions_usg_df = self.lesions_usg_df.drop(
            columns=["mask other filename", "pixel size", "verification"]
        )

        # 2. Adjust the birads values, folding known-biopsy-proven (6) into 5 as
        # every other adapter does
        self.lesions_usg_df["birads"] = self.lesions_usg_df["birads"].apply(
            lambda x: collapse_known_biopsy_proven(
                map_birads(x, source="Breast-Lesions-USG", modality=ULTRASOUND)
            )
        )

        # 3. replace not applicable with None, and not available with None
        self.lesions_usg_df = self.lesions_usg_df.replace("not applicable", None)
        self.lesions_usg_df = self.lesions_usg_df.replace("not available", None)

        self.run_row_pool(self.lesions_usg_df)

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
            patient_context=PatientContext(age=bin_age(row["age"])),
            exam_context=ExamContext(modality=ULTRASOUND),
        )

        # construct final exam info
        assessment_birads = get_proper_birads(row["birads"], modality=ULTRASOUND)

        findings = JSONReportFindings.build(
            breast=BreastFindings(
                tissue_composition=get_proper_tissue_composition(
                    row["tissue composition"]
                )
            ),
            lesion=(
                self._construct_lesion(row)
                if not isna_v2(row["mask tumor filename"])
                else NOT_PRESENT if row["birads"] == BIRADS_MAPPING[1] else None
            ),
            assessment=Assessment(birads=assessment_birads),
            # One record per imaged lesion, with its own mask.
            annotation_scope=SCOPE_COMPLETE,
        )

        return ExamInformation(
            id=exam_id,
            patient=patient_id,
            dataset=self.get_dataset_name(),
            modality=ULTRASOUND,
            birads=assessment_birads,
            race=self._race,
            machine=self._machine,
            exam=exam_imgs_path,
            segmentation=(
                [segmentation_path] if not isna_v2(segmentation_path) else None
            ),
            context=context.get_string(),
            findings=findings.get_string(),
            exam_id=patient_id,
            breast_id=None,
            assessment_scope=ASSESSMENT_GROUNDED,
        )

    def _construct_lesion(self, row) -> Lesion:
        return Lesion(
            finding_type=MASS,
            mass_details=MassAbnormality(
                shape=get_proper_mass_shape(row["shape"]),
                margin=get_proper_mass_margin(row["margin"], modality=ULTRASOUND),
                echo_pattern=get_proper_mass_echo_pattern(row["echogenicity"]),
                posterior_features=get_proper_posterior_features(
                    row["posterior features"]
                ),
                echogenic_feature=get_proper_echogenic_feature(row["halo"]),
            ),
        )
