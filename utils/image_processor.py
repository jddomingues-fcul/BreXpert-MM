import logging
import os
from typing import Optional

import numpy as np

from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from utils.error_handling import log_func_info, trycatch_func


class ImageProcessor:
    IMGS_SUFFIX = "_imgs"
    SEGMENTATION_SUFFIX = "_segmentation"
    SLICES_SUFFIX = "_slices"

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

        # Create the processed imgs folder
        os.makedirs(self.processed_imgs_path, exist_ok=True)

    @trycatch_func
    @log_func_info
    def read_image(self, img_path: str):
        assert os.path.exists(img_path), f"Image {img_path} not found"
        res = None

        try:
            res = self.read_process(img_path)
        except Exception as e:
            logging.error(f"Error reading the image {img_path}: {e}")

        return res

    @trycatch_func
    @log_func_info
    def apply_processing(self, img: np.ndarray, is_segmentation: bool = False):
        res = img
        try:
            if is_segmentation:
                for processing_step in self.segmentation_pipeline:
                    res = processing_step(res)
            else:
                for processing_step in self.pipeline:
                    res = processing_step(res)
        except Exception as e:
            logging.error(f"Error processing image: {e}")
            return None

        return res

    @trycatch_func
    @log_func_info
    def process_image(self, img_path: str):
        assert os.path.exists(img_path), f"Image {img_path} not found"
        res = None

        try:
            res = self.read_process(img_path)
            for processing_step in self.pipeline:
                res = processing_step(res)
        except Exception as e:
            logging.error(f"Error processing image {img_path}: {e}")

        return res

    @trycatch_func
    @log_func_info
    def process_and_save_image(self, img_path: str, exam_id: str, patient_id: str):
        res = self.process_image(img_path)

        if res is None:
            return None

        save_path = os.path.join(
            self.processed_imgs_path, f"{patient_id}-{exam_id}{self.IMGS_SUFFIX}"
        )
        return self.save_process(save_path, [res])

    @trycatch_func
    @log_func_info
    def save_image_set(self, imgs: list, exam_id: str, patient_id: str):
        save_path = os.path.join(
            self.processed_imgs_path, f"{patient_id}-{exam_id}{self.IMGS_SUFFIX}"
        )
        return self.save_process(save_path, imgs)

    @trycatch_func
    @log_func_info
    def save_segmentation_set(self, segs: list, exam_id: str, patient_id: str):
        save_path = os.path.join(
            self.processed_imgs_path,
            f"{patient_id}-{exam_id}{self.SEGMENTATION_SUFFIX}",
        )
        return self.save_process(save_path, segs)

    @trycatch_func
    @log_func_info
    def process_segmentation_mask(self, segmentation_path: Optional[str]):
        if segmentation_path is None:
            return None

        if not os.path.exists(segmentation_path):
            return None

        seg = None

        try:
            seg = self.read_process(segmentation_path)
            for processing_step in self.segmentation_pipeline:
                seg = processing_step(seg)
            seg = seg > 0
            if seg is None or seg.max() == 0:  # type: ignore
                return None
        except Exception as e:
            logging.error(
                f"Error processing segmentation mask {segmentation_path}: {e}"
            )

        return seg

    @trycatch_func
    @log_func_info
    def save_segmentation_mask(
        self, segmentation_path: Optional[str], exam_id: str, patient_id: str
    ):
        seg = self.process_segmentation_mask(segmentation_path)
        if seg is None:
            return None

        save_path = os.path.join(
            self.processed_imgs_path,
            f"{patient_id}-{exam_id}{self.SEGMENTATION_SUFFIX}",
        )
        return self.save_process(save_path, [seg])
