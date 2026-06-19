import os
import uuid
from functools import partial

from tqdm import tqdm

from dtos.breast_cancer_dataset import (
    BreastCancerDataset,
    ExamInformation,
)
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext
from dtos.json_report_findings import Assessment, JSONReportFindings
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    NOT_PRESENT,
    UNKNOWN,
    birads_mapping,
    column_cleaning_csv_reading,
    get_proper_birads,
    isna_v2,
    read_breast_image,
)

raw_imgs_path = "../data/raw/busbra/imgs"
raw_segs_path = "../data/raw/busbra/masks"
raw_imgs_extension = ".png"
data_csv = "../data/raw/busbra/bus_data.csv"
processed_imgs_path = "../data/processed/busbra/imgs"
csv_save_path = "../data/processed/busbra/busbra.csv"


class BUSBRA(BreastCancerDataset):
    _modality = "us"
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

        self.data_df = column_cleaning_csv_reading(data_csv)

    def process_info(self):
        # 1. Adjust the birads values
        self.data_df["birads"] = self.data_df["birads"].apply(
            lambda x: birads_mapping.get(x, UNKNOWN)
        )

        # 2. The single side values are ambiguous, so we set them to NaN since we don't know effectively which side they refer to
        self.data_df["side"] = self.data_df["side"].replace("single", None)

        for exam in tqdm(
            self.data_df.itertuples(index=False),
            total=len(self.data_df),
            desc=f"Processing {self.get_dataset_name()} dataset",
            unit="exams",
            ncols=100,
        ):
            exam_info = self.process_example(exam)
            if exam_info is not None:
                self.append_exam(exam_info)

    def process_example(self, row):
        # get needed info for exam
        exam_id = str(uuid.uuid4())
        patient_id = f"{self.get_dataset_name()}-{row.case}"

        exam_imgs_path = self.image_processor.process_and_save_image(
            os.path.join(raw_imgs_path, row.id + raw_imgs_extension),
            exam_id,
            patient_id,
        )

        segmentation_path = self.image_processor.save_segmentation_mask(
            os.path.join(
                raw_segs_path, row.id.replace("bus", "mask") + raw_imgs_extension
            ),
            exam_id,
            patient_id,
        )

        context = JSONReportContext(
            exam_context=ExamContext(modality=self._modality, laterality=row.side)
        )
        findings = JSONReportFindings(
            lesion=NOT_PRESENT if row.birads == birads_mapping[1] else None,
            assessment=Assessment(birads=get_proper_birads(row.birads)),
        )

        return ExamInformation(
            id=exam_id,
            patient=patient_id,
            dataset=self.get_dataset_name(),
            modality=self._modality,
            birads=get_proper_birads(row.birads),
            race=self._race,
            machine=row.device,
            exam=exam_imgs_path,
            segmentation=(
                [segmentation_path] if not isna_v2(segmentation_path) else None
            ),
            context=context.get_string(),
            findings=findings.get_string(),
        )
