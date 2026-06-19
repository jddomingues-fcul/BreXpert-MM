import logging
import uuid
from dataclasses import dataclass
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from dtos.breast_cancer_dataset import (
    BreastCancerDataset,
    ExamInformation,
)
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext
from dtos.json_report_findings import Assessment, JSONReportFindings
from utils.error_handling import trycatch_func
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    NOT_PRESENT,
    UNKNOWN,
    birads_assessment,
    birads_mapping,
    get_proper_birads,
    get_value,
    isna_v2,
    process_us_from_mat,
    read_mat_images,
)

raw_data_path = "../data/raw/oasbud/OASBUD.mat"
processed_imgs_path = "../data/processed/oasbud/imgs"
csv_save_path = "../data/processed/oasbud/oasbud.csv"


@dataclass(frozen=True)
class USDataHolder:
    patient_id: str
    img: np.ndarray
    segmentation: np.ndarray
    birads: str


class Oasbud(BreastCancerDataset):
    _modality = "us"
    _machine = UNKNOWN
    _race = UNKNOWN

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path="",
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(
                read_func=partial(read_mat_images)
            ),  # NOTE: this case, is very specific and needs to be manually handled in the dataset class,
        )

        self.source_data = self.image_processor.read_image(raw_data_path)

    def process_info(self):
        acc_data: list[USDataHolder] = []
        assert self.source_data is not None, "No data found in the dataset"

        for i, exam_set in tqdm(
            enumerate(self.source_data), desc="Processing examples oasbud"
        ):
            _, us1, us2, seg1, seg2, birads, _ = exam_set
            birads = birads[0]
            us1, us2 = process_us_from_mat(us1), process_us_from_mat(us2)
            birads = get_value(birads, birads_assessment)
            if birads is None:
                continue

            acc_data.append(
                USDataHolder(
                    patient_id=str(i),
                    img=us1,
                    segmentation=seg1,
                    birads=birads,
                )
            )
            acc_data.append(
                USDataHolder(
                    patient_id=str(i),
                    img=us2,
                    segmentation=seg2,
                    birads=birads,
                )
            )

        # Process the exams
        n = cpu_count() - 1
        with Pool(processes=n) as p:
            results = p.map(self.process_example, acc_data)

        for result in results:
            if not isna_v2(result):
                self.append_exam(result)

    @trycatch_func
    def process_example(self, example: USDataHolder):
        # get needed info for exam
        exam_id = str(uuid.uuid4())
        patient_id = f"{self.get_dataset_name()}-{example.patient_id}"

        exam, seg = self.image_processor.apply_processing(
            example.img, False
        ), self.image_processor.apply_processing(example.segmentation, True)
        exam_path = self.image_processor.save_image_set([exam], exam_id, patient_id)

        if isna_v2(exam):
            logging.warning(
                f"Skipping exam for patient {patient_id} due to processing issues."
            )
            return None

        segmentation_path = self.image_processor.save_segmentation_set(
            [seg], exam_id, patient_id
        )

        # construct context
        context = JSONReportContext(exam_context=ExamContext(modality=self._modality))

        # construct final exam info
        findings = JSONReportFindings(
            lesion=NOT_PRESENT if example.birads == birads_mapping[1] else None,
            assessment=Assessment(birads=get_proper_birads(example.birads)),
        )

        return ExamInformation(
            id=exam_id,
            patient=patient_id,
            dataset=self.get_dataset_name(),
            modality=self._modality,
            birads=get_proper_birads(example.birads),
            race=self._race,
            machine=self._machine,
            exam=exam_path,
            segmentation=(
                [segmentation_path] if not isna_v2(segmentation_path) else None
            ),
            context=context.get_string(),
            findings=findings.get_string(),
        )
