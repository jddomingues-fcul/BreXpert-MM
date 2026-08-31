from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

import numpy as np

from utils.preprocessing import (
    MAMMOGRAPHY,
    MRI,
    ULTRASOUND,
    binarize_mask,
    convert_dcm_image,
    fit_mask_to_canvas,
    fit_to_canvas,
    save_images_as_npy,
)

MAMMOGRAPHY_CANVAS = (784, 1120)
ULTRASOUND_CANVAS = (672, 560)
MRI_CANVAS = (560, 560)

CANVAS_BY_MODALITY: dict[str, tuple[int, int]] = {
    MAMMOGRAPHY: MAMMOGRAPHY_CANVAS,
    ULTRASOUND: ULTRASOUND_CANVAS,
    MRI: MRI_CANVAS,
}


def canvas_for(modality: str) -> tuple[int, int]:
    if modality not in CANVAS_BY_MODALITY:
        raise KeyError(
            f"No stored-image canvas defined for modality {modality!r}; "
            f"known modalities: {sorted(CANVAS_BY_MODALITY)}"
        )
    return CANVAS_BY_MODALITY[modality]


@dataclass(frozen=True)
class ImagePreprocessingConfig:
    modality: str = MAMMOGRAPHY
    read_func: Callable = partial(convert_dcm_image)
    save_func: Callable = partial(save_images_as_npy, dtype=np.uint16)
    processing_pipeline: list[Callable] | None = None
    segmentation_pipeline: list[Callable] | None = None

    # Resolved geometry, filled in by __post_init__ so callers can read it back.
    canvas: tuple[int, int] = field(default=(0, 0))

    def __post_init__(self):
        canvas = self.canvas if self.canvas != (0, 0) else canvas_for(self.modality)
        object.__setattr__(self, "canvas", canvas)

        if self.processing_pipeline is None:
            object.__setattr__(
                self,
                "processing_pipeline",
                [partial(fit_to_canvas, canvas=canvas)],
            )
        if self.segmentation_pipeline is None:
            object.__setattr__(
                self,
                "segmentation_pipeline",
                [binarize_mask, partial(fit_mask_to_canvas, canvas=canvas)],
            )
