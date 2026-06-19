from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import numpy as np

from utils.preprocessing import (
    convert_dcm_image,
    pad_to_largest_dim,
    resize_breast_image,
    save_images_as_npy,
)

RESIZE_DIMS = (512, 512)


@dataclass(frozen=True)
class ImagePreprocessingConfig:
    read_func: Callable = partial(convert_dcm_image)
    save_func: Callable = partial(save_images_as_npy, dtype=np.uint16)
    processing_pipeline: list[Callable] = field(
        default_factory=lambda: [
            pad_to_largest_dim,
            partial(resize_breast_image, resize_value=RESIZE_DIMS),
        ]
    )
    segmentation_pipeline: list[Callable] = field(
        default_factory=lambda: [
            pad_to_largest_dim,
            partial(resize_breast_image, resize_value=RESIZE_DIMS),
        ]
    )
