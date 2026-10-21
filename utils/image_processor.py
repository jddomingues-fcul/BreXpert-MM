import logging
import os

import numpy as np

from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from utils.error_handling import log_func_info, trycatch_func
from utils.preprocessing import binarize_mask


class ImageProcessor:
    """Read -> process -> save, for one modality's images and masks.

    Every public method is wrapped in `trycatch_func`, so a failure anywhere
    below surfaces as `None` with the cause logged. Callers only need to check
    for `None`.
    """

    IMGS_SUFFIX = "_imgs"
    SEGMENTATION_SUFFIX = "_segmentation"

    def __init__(
        self,
        raw_imgs_path: str,
        processed_imgs_path: str,
        image_preprocessing_config: ImagePreprocessingConfig,
    ):
        self.raw_imgs_path = raw_imgs_path
        self.processed_imgs_path = processed_imgs_path
        self.read_process = image_preprocessing_config.read_func
        self.save_process = image_preprocessing_config.save_func
        self.pipeline = image_preprocessing_config.processing_pipeline
        self.segmentation_pipeline = image_preprocessing_config.segmentation_pipeline

        os.makedirs(self.processed_imgs_path, exist_ok=True)

    def _save_path(self, exam_id: str, patient_id: str, suffix: str) -> str:
        return os.path.join(self.processed_imgs_path, f"{patient_id}-{exam_id}{suffix}")

    @trycatch_func
    @log_func_info
    def read_image(self, img_path: str):
        return self.read_process(img_path)

    @trycatch_func
    @log_func_info
    def apply_processing(self, img: np.ndarray, is_segmentation: bool = False):
        pipeline = self.segmentation_pipeline if is_segmentation else self.pipeline
        for processing_step in pipeline:
            img = processing_step(img)
        return img

    @trycatch_func
    @log_func_info
    def process_image(self, img_path: str):
        return self.apply_processing(self.read_process(img_path))

    @trycatch_func
    @log_func_info
    def process_and_save_image(self, img_path: str, exam_id: str, patient_id: str):
        img = self.process_image(img_path)
        if img is None:
            return None
        return self.save_process(
            self._save_path(exam_id, patient_id, self.IMGS_SUFFIX), [img]
        )

    @trycatch_func
    @log_func_info
    def save_image_set(self, imgs: list, exam_id: str, patient_id: str):
        if not imgs or any(img is None for img in imgs):
            logging.warning(
                f"Not saving image set for patient {patient_id}, exam {exam_id}: missing image data."
            )
            return None

        return self.save_process(
            self._save_path(exam_id, patient_id, self.IMGS_SUFFIX), imgs
        )

    @trycatch_func
    @log_func_info
    def save_segmentation_set(self, segs: list, exam_id: str, patient_id: str):
        if not segs or any(seg is None for seg in segs):
            return None

        segs = [binarize_mask(seg) for seg in segs]
        if all(seg.max() == 0 for seg in segs):
            logging.warning(
                f"Empty segmentation for patient {patient_id}, exam {exam_id}. Not saving."
            )
            return None

        return self.save_process(
            self._save_path(exam_id, patient_id, self.SEGMENTATION_SUFFIX), segs
        )

    @trycatch_func
    @log_func_info
    def process_segmentation_mask(self, segmentation_path: str | None):
        if segmentation_path is None or not os.path.exists(segmentation_path):
            return None

        seg = self.apply_processing(
            self.read_process(segmentation_path), is_segmentation=True
        )
        if seg is None:
            return None

        # An all-zero mask carries no annotation; treat it as no mask at all.
        seg = binarize_mask(seg)
        return None if seg.max() == 0 else seg

    @trycatch_func
    @log_func_info
    def save_segmentation_mask(
        self, segmentation_path: str | None, exam_id: str, patient_id: str
    ):
        seg = self.process_segmentation_mask(segmentation_path)
        if seg is None:
            return None

        return self.save_process(
            self._save_path(exam_id, patient_id, self.SEGMENTATION_SUFFIX), [seg]
        )
