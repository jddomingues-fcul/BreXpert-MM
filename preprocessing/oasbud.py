import logging
from dataclasses import dataclass
from functools import partial

import numpy as np
from tqdm import tqdm

from dtos.breast_cancer_dataset import (
    ASSESSMENT_GROUNDED,
    BreastCancerDataset,
    ExamInformation,
    stable_exam_id,
)
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext
from dtos.json_report_findings import (
    SCOPE_COMPLETE,
    Assessment,
    JSONReportFindings,
)
from utils.error_handling import pool_worker
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    BIRADS_MAPPING,
    NOT_PRESENT,
    PRESENT,
    ULTRASOUND,
    UNKNOWN,
    get_proper_birads,
    isna_v2,
    map_birads,
    process_us_from_mat,
    read_mat_images,
)

raw_data_path = "../data/raw/oasbud/OASBUD.mat"
processed_imgs_path = "../data/processed/oasbud/imgs"
csv_save_path = "../data/processed/oasbud/oasbud.csv"


@dataclass(frozen=True)
class USDataHolder:
    patient_id: str
    view: int
    img: np.ndarray
    segmentation: np.ndarray
    birads: str


class Oasbud(BreastCancerDataset):
    _modality = ULTRASOUND
    _machine = UNKNOWN
    _race = UNKNOWN

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path="",
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(
                modality=ULTRASOUND, read_func=partial(read_mat_images)
            ),  # NOTE: this case, is very specific and needs to be manually handled in the dataset class,
        )

        self.source_data = self.image_processor.read_image(raw_data_path)

    def process_info(self):
        acc_data: list[USDataHolder] = []
        assert self.source_data is not None, "No data found in the dataset"
        self.report.rows_seen = len(self.source_data)

        for i, exam_set in tqdm(
            enumerate(self.source_data), desc="Processing examples oasbud"
        ):
            record_id, us1, us2, seg1, seg2, birads, _ = exam_set
            birads = birads[0]
            us1, us2 = process_us_from_mat(us1), process_us_from_mat(us2)
            birads = map_birads(birads, source="OASBUD", modality=ULTRASOUND)
            if birads is None:
                self.report.filtered_rows += 1
                continue

            patient_id = self.extract_patient_id(record_id, i)

            acc_data.append(
                USDataHolder(
                    patient_id=patient_id,
                    view=1,
                    img=us1,
                    segmentation=seg1,
                    birads=birads,
                )
            )
            acc_data.append(
                USDataHolder(
                    patient_id=patient_id,
                    view=2,
                    img=us2,
                    segmentation=seg2,
                    birads=birads,
                )
            )

        self.run_pool(self.process_example, acc_data, "Processing oasbud examples")

    @staticmethod
    def extract_patient_id(record_id, fallback_index: int) -> str:
        try:
            value = np.asarray(record_id).ravel()
            if value.size == 0:
                return f"lesion-{fallback_index}"
            text = str(value[0]).strip()
            return text if text else f"lesion-{fallback_index}"
        except Exception:
            return f"lesion-{fallback_index}"

    @pool_worker
    def process_example(self, example: USDataHolder):
        # get needed info for exam
        exam_id = stable_exam_id(
            self.get_dataset_name(), example.patient_id, example.view
        )
        patient_id = f"{self.get_dataset_name()}-{example.patient_id}"

        exam, seg = self.image_processor.apply_processing(
            example.img, False
        ), self.image_processor.apply_processing(example.segmentation, True)

        if isna_v2(exam):
            logging.warning(
                f"Skipping exam for patient {patient_id} due to processing issues."
            )
            return None

        exam_path = self.image_processor.save_image_set([exam], exam_id, patient_id)
        if isna_v2(exam_path):
            return None

        segmentation_path = self.image_processor.save_segmentation_set(
            [seg], exam_id, patient_id
        )

        # construct context
        context = JSONReportContext(exam_context=ExamContext(modality=self._modality))

        # construct final exam info
        # A saved segmentation is positive evidence that a lesion is in the frame,
        # even though OASBUD ships no descriptors for it; leaving the field unset
        # would say we do not know.
        if example.birads == BIRADS_MAPPING[1]:
            lesion = NOT_PRESENT
        elif not isna_v2(segmentation_path):
            lesion = PRESENT
        else:
            lesion = None

        assessment_birads = get_proper_birads(example.birads, modality=ULTRASOUND)

        findings = JSONReportFindings.build(
            lesion=lesion,
            assessment=Assessment(birads=assessment_birads),
            annotation_scope=SCOPE_COMPLETE,
        )

        return [
            ExamInformation(
                id=exam_id,
                patient=patient_id,
                dataset=self.get_dataset_name(),
                modality=self._modality,
                birads=assessment_birads,
                race=self._race,
                machine=self._machine,
                exam=exam_path,
                segmentation=(
                    [segmentation_path] if not isna_v2(segmentation_path) else None
                ),
                context=context.get_string(),
                findings=findings.get_string(),
                exam_id=patient_id,
                breast_id=None,
                assessment_scope=ASSESSMENT_GROUNDED,
            )
        ]
