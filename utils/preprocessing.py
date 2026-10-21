from __future__ import annotations

import json
import logging
import numbers
import os
import re
from collections import Counter
from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pydicom as dicom
import scipy
import SimpleITK as sitk
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut
from scipy.signal import hilbert

from utils.error_handling import log_func_info, trycatch_func

logger = logging.getLogger(__name__)

LEXICON_VERSION = "v2025"

MAMMOGRAPHY = "mammography"
ULTRASOUND = "ultrasound"
MRI = "mri"
MODALITIES = (MAMMOGRAPHY, ULTRASOUND, MRI)

# Short codes. These are what the split filenames and the per-label quota tables
# use; keeping the mapping here stops the two from drifting apart again.
MODALITY_SHORT = {MAMMOGRAPHY: "mg", ULTRASOUND: "us", MRI: "mr"}

UNKNOWN = "unknown"
NOT_PRESENT = "not_present"
PRESENT = "present"
BILATERAL = "bilateral"
CRANIAL_CAUDAL = "cranial caudal"
MEDIOLATERAL_OBLIQUE = "mediolateral oblique"

_UNMAPPED_COUNTS: Counter = Counter()


def warn_unmapped(field: str, value: Any, consequence: str = "returning None") -> None:
    key = (field, str(value))
    seen = _UNMAPPED_COUNTS[key]
    _UNMAPPED_COUNTS[key] += 1
    if seen == 0:
        logger.warning(f"{field}: value {value!r} not recognized, {consequence}.")


def drain_unmapped_counts() -> dict[tuple[str, str], int]:
    """Take and clear this process's unmapped-value tally.

    Adapters run their rows through a multiprocessing Pool, and each worker gets
    its own copy of this module. Draining in the worker and returning the tally
    with the results is the only way the parent ever sees it -- otherwise the
    counts die with the worker and the report is silently empty.
    """
    counts = dict(_UNMAPPED_COUNTS)
    _UNMAPPED_COUNTS.clear()
    return counts


def merge_unmapped_counts(target: Counter, counts) -> Counter:
    """Fold a worker's drained tally back into the parent's."""
    if not counts:
        return target
    target.update(dict(counts))
    return target


def format_unmapped_report(counts, limit: int = 40) -> str:
    """A human-readable 'what did the lexicon fail to map' table."""
    if not counts:
        return "No unmapped values."

    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    total = sum(counts.values())
    lines = [
        f"{total} unmapped value occurrences across {len(ordered)} distinct "
        f"(field, value) pairs:"
    ]
    for (field, value), count in ordered[:limit]:
        lines.append(f"  {count:>7,}  {field}: {value!r}")
    if len(ordered) > limit:
        lines.append(f"  ... and {len(ordered) - limit} more distinct pairs")
    return "\n".join(lines)


def normalize_to_uint16(
    img: np.ndarray,
    value_range: tuple[float, float] | None = None,
) -> np.ndarray:
    img = np.asarray(img, dtype=np.float32)

    if value_range is None:
        img_min, img_max = float(np.nanmin(img)), float(np.nanmax(img))
    else:
        img_min, img_max = float(value_range[0]), float(value_range[1])

    if not np.isfinite(img_min) or not np.isfinite(img_max) or img_max <= img_min:
        return np.zeros(img.shape, dtype=np.uint16)

    scaled = np.clip((img - img_min) / (img_max - img_min), 0.0, 1.0)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=0.0)
    return np.rint(scaled * 65535.0).astype(np.uint16)


def apply_dicom_photometric_transforms(ds, data: np.ndarray) -> np.ndarray:
    try:
        data = apply_modality_lut(data, ds)
    except Exception as e:  # LUT is optional and malformed ones are common
        logger.debug(f"Modality LUT not applied: {e}")

    try:
        data = apply_voi_lut(data, ds, prefer_lut=True)
    except Exception as e:
        logger.debug(f"VOI LUT not applied: {e}")

    photometric = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
    if photometric == "MONOCHROME1":
        data = np.asarray(data)
        data = data.max() - data

    return data


@trycatch_func
@log_func_info
def read_breast_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Image {img_path} could not be read at `read_breast_image`")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return normalize_to_uint16(img)


@trycatch_func
@log_func_info
def read_nii_gz_images(img_path: str) -> np.ndarray:
    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    img_np = sitk.GetArrayFromImage(img)
    return normalize_to_uint16(img_np)


@trycatch_func
@log_func_info
def read_nii_gz_raw(img_path: str) -> np.ndarray:
    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    return sitk.GetArrayFromImage(img)


@trycatch_func
@log_func_info
# NOTE: specific to OASBUD dataset, not need to norm here as it is done in process_us_from_mat
def read_mat_images(img_path: str) -> np.ndarray:
    mat = scipy.io.loadmat(img_path)
    res = mat["data"][0]
    return res


@trycatch_func
@log_func_info
def process_us_from_mat(
    img: np.ndarray, db_threshold: int = -50, axis: int = 0
) -> np.ndarray:
    """Envelope-detect and log-compress raw RF ultrasound into a B-mode image.

    `axis` is the axial (depth) direction, which is where the transducer carrier
    lives -- Hilbert demodulation along any other axis does not recover an
    envelope. RF frames are stored (samples, lines), so depth is axis 0.
    """
    img = np.asarray(img, dtype=np.float32)
    if img.ndim >= 2 and img.shape[axis] < max(img.shape):
        logger.warning(
            f"process_us_from_mat: demodulating along axis {axis} of a {img.shape} "
            "frame, which is not the longest axis. Depth should be the long axis for "
            "RF data -- verify the orientation before trusting these B-mode images."
        )

    envelope_im = np.abs(hilbert(img, axis=axis))  # type: ignore
    peak = float(np.max(envelope_im))
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("RF frame has no signal; cannot log-compress a zero envelope.")

    # Floor the envelope at the display threshold before the log so that silent
    # samples give -inf-free values.
    floor = peak * (10.0 ** (db_threshold / 20.0))
    compress_im = 20 * np.log10(np.maximum(envelope_im, floor) / peak)
    result = compress_im.astype("float32")
    return normalize_to_uint16(result, value_range=(float(db_threshold), 0.0))


@trycatch_func
@log_func_info
def load_images_from_npy(npy_path: str) -> np.ndarray:
    if not str(npy_path).endswith(".npy"):
        raise ValueError(f"Invalid file extension for {npy_path}")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"File {npy_path} not found")
    return np.load(npy_path)


def _resize_interpolation(src_shape: tuple, dst_wh: tuple) -> int:
    """INTER_AREA when shrinking, INTER_LINEAR when growing.

    cv2.resize defaults to INTER_LINEAR, which aliases badly on a large
    downscale: a mammogram going from ~2400x4900 to a few hundred pixels loses
    roughly a decade of scale, and bilinear sampling of a handful of source
    pixels per output pixel turns fine texture into noise. INTER_AREA averages
    over the whole source footprint, which is what a decimation needs.
    """
    src_h, src_w = src_shape[:2]
    dst_w, dst_h = dst_wh
    return cv2.INTER_AREA if (dst_w < src_w or dst_h < src_h) else cv2.INTER_LINEAR


def binarize_mask(mask: np.ndarray) -> np.ndarray:
    return (np.asarray(mask) > 0).astype(np.uint8)


@trycatch_func
@log_func_info
def resize_mask(mask: np.ndarray, resize_value: tuple) -> np.ndarray:
    return cv2.resize(
        binarize_mask(mask), resize_value, interpolation=cv2.INTER_NEAREST
    )


def _letterbox(
    image: np.ndarray, canvas: tuple[int, int], interpolation: int
) -> np.ndarray:
    """Scale to fit inside `canvas` preserving aspect, then centre-pad."""
    if image.ndim != 2:
        raise ValueError(
            f"fit_to_canvas expects a 2D image, got shape {image.shape}. Select a "
            "slice/frame/channel in the adapter before processing."
        )

    canvas_w, canvas_h = canvas
    src_h, src_w = image.shape
    if src_h == 0 or src_w == 0:
        raise ValueError(f"Cannot fit an empty image of shape {image.shape}.")

    scale = min(canvas_w / src_w, canvas_h / src_h)
    new_w = max(1, min(canvas_w, int(round(src_w * scale))))
    new_h = max(1, min(canvas_h, int(round(src_h * scale))))

    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_w, pad_h = canvas_w - new_w, canvas_h - new_h
    left, top = pad_w // 2, pad_h // 2
    return cv2.copyMakeBorder(
        resized,
        top,
        pad_h - top,
        left,
        pad_w - left,
        cv2.BORDER_CONSTANT,
        value=0,
    )


@trycatch_func
@log_func_info
def fit_to_canvas(image: np.ndarray, canvas: tuple[int, int]) -> np.ndarray:
    """Letterbox an image onto a fixed (width, height) canvas.

    Replaces the old pad-to-square-then-resize pipeline. That combination was
    not wrong in itself -- padding to a square and then resizing to a square
    target gives the same result as scaling first -- but the *square target* is
    what cost the resolution: a 1:2 portrait mammogram only ever filled half of
    a square canvas, so half the stored pixels were padding and the breast was
    reconstructed at half the available width. A canvas whose aspect matches the
    modality keeps the same pixel budget on tissue instead of on background.
    """
    return _letterbox(image, canvas, _resize_interpolation(image.shape, canvas))


@trycatch_func
@log_func_info
def fit_mask_to_canvas(mask: np.ndarray, canvas: tuple[int, int]) -> np.ndarray:
    """`fit_to_canvas` for a binary mask: nearest neighbour, re-binarised.

    Any interpolation other than nearest invents intermediate values that a
    later `> 0` test promotes to foreground, which silently dilates the
    annotation.
    """
    return binarize_mask(_letterbox(binarize_mask(mask), canvas, cv2.INTER_NEAREST))


@trycatch_func
@log_func_info
def save_images_as_npy(save_path: str, images: list, dtype=None) -> str:
    """Save one or more processed frames.

    A single frame is stored as a plain 2D array. Previously every call wrapped
    its frame in a list, so each file carried a useless leading axis of length 1
    that every reader then had to index away.
    """
    if not str(save_path).endswith(".npy"):
        save_path = f"{save_path}.npy"

    arr = np.array(images, dtype=dtype) if dtype else np.array(images)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]

    np.save(save_path, arr)
    return save_path


@trycatch_func
@log_func_info
def load_image_frames(npy_path: str) -> np.ndarray:
    """Load a saved image as a (frames, H, W) stack regardless of how it was written."""
    arr = load_images_from_npy(npy_path)
    if arr is None:
        raise ValueError(f"Could not load {npy_path}")
    return arr[None, ...] if arr.ndim == 2 else arr


@trycatch_func
@log_func_info
def convert_dcm_image(img_path: str) -> np.ndarray:
    ds = dicom.dcmread(img_path)
    data = ds.pixel_array

    samples_per_pixel = int(getattr(ds, "SamplesPerPixel", 1) or 1)
    number_of_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
    photometric = str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()

    if photometric.startswith("PALETTE"):
        raise ValueError(
            f"{img_path} is PALETTE COLOR: its stored values are colour-table indices, "
            "not intensities, so reading them as greyscale is meaningless. Such series "
            "are derived colour maps and must be excluded in the adapter."
        )

    if number_of_frames > 1:
        raise ValueError(
            f"{img_path} is a {number_of_frames}-frame series; this pipeline stores "
            "single 2D images. Select a frame in the adapter before converting."
        )

    if samples_per_pixel > 1:
        data = cv2.cvtColor(np.asarray(data), cv2.COLOR_RGB2GRAY)

    data = apply_dicom_photometric_transforms(ds, data)

    if data.max() == data.min():
        raise ValueError(
            f"The DCM image {img_path} has no variation in pixel values (max == min)."
        )

    return normalize_to_uint16(data)


def _is_box(roi: Any) -> bool:
    """A box is four numbers: (ymin, xmin, ymax, xmax)."""
    if isinstance(roi, (str, bytes)) or not hasattr(roi, "__len__"):
        return False
    return len(roi) == 4 and all(
        isinstance(v, numbers.Real) and not isinstance(v, bool) for v in roi
    )


@trycatch_func
@log_func_info
def create_segmentation_mask_aux(mask: np.ndarray, rois):
    height, width = mask.shape

    for roi in rois:
        if _is_box(roi):
            ymin, xmin, ymax, xmax = (round(float(v)) for v in roi)
            ymin, ymax = max(0, min(ymin, ymax)), min(height, max(ymin, ymax))
            xmin, xmax = max(0, min(xmin, xmax)), min(width, max(xmin, xmax))
            if ymax > ymin and xmax > xmin:
                mask[ymin:ymax, xmin:xmax] = 1
            else:
                logger.warning(f"ROI {tuple(roi)} lies outside the {mask.shape} frame.")
            continue

        if hasattr(roi, "__iter__") and not isinstance(roi, (str, bytes)):
            create_segmentation_mask_aux(mask, roi)
            continue

        logger.warning(f"Skipping ROI entry {roi!r}: not a box and not iterable.")

    return mask


@trycatch_func
@log_func_info
def create_segmentation_mask(
    height: int, width: int, rois: tuple[tuple[int, int, int, int]]
) -> np.ndarray:
    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    return create_segmentation_mask_aux(mask, rois)


@trycatch_func
def upper_to_lower_wspace(s):
    # This regex matches positions where an uppercase letter is followed by a lowercase letter
    # or where a lowercase letter is followed by an uppercase letter.
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", s).lower()


def bin_age(value: Any) -> str | None:
    """Coarse age band, or None if the value is missing or not a plausible age.

    Ages are only ever published in bands, so parsing, range-checking and
    binning all happen here rather than being split across two helpers.
    """
    if isna_v2(value):
        return None

    try:
        age = int(float(value))
    except (ValueError, TypeError):
        warn_unmapped("age", value, "not convertible to an integer for binning")
        return None

    if not 0 <= age <= 120:
        warn_unmapped("age", value, "outside the plausible 0-120 range")
        return None

    if age <= 39:
        return "<40"
    if age < 50:
        return "40-49"
    if age < 60:
        return "50-59"
    if age < 70:
        return "60-69"
    if age < 80:
        return "70-79"
    return "80+"


def get_proper_contrast_phase(value) -> str | None:
    if isna_v2(value):
        return None

    x = normalize_lexicon_value(value)

    contrast_phase_mapping = {
        "apparent diffusion coefficient image": "adc",
        "diffusion image": "diffusion",
        "t1 with no fat saturation": "t1",
        "t2 with no fat saturation": "t2",
        "pre-contrast": "pre-contrast",
        "pre contrast phase - axial view": "pre-contrast",
        "pre-contrast t1 fat saturated dynamic": "pre-contrast",
        "first post-contrast phase sequence": "post-contrast phase 1",
        "first postcontrast t1 fat saturated dynamics": "post-contrast phase 1",
        "registered ax sen vibrant multi phase - contrast phase: 1": "post-contrast phase 1",
        "contrast phase #1 - axial view": "post-contrast phase 1",
        "post-contrast phase 1": "post-contrast phase 1",
        "second post-contrast phase sequence": "post-contrast phase 2",
        "second postcontrast t1 fat saturated dynamics": "post-contrast phase 2",
        "registered ax sen vibrant multi phase - contrast phase: 2": "post-contrast phase 2",
        "contrast phase #2 - axial view": "post-contrast phase 2",
        "post-contrast phase 2": "post-contrast phase 2",
        "third post-contrast phase sequence": "post-contrast phase 3",
        "third postcontrast t1 fat saturated dynamics": "post-contrast phase 3",
        "registered ax sen vibrant multi phase - contrast phase: 3": "post-contrast phase 3",
        "contrast phase #3 - axial view": "post-contrast phase 3",
        "post-contrast phase 3": "post-contrast phase 3",
        "fourth post-contrast phase sequence": "post-contrast phase 4",
        "fourth postcontrast t1 fat saturated dynamics": "post-contrast phase 4",
        "registered ax sen vibrant multi phase - contrast phase: 4": "post-contrast phase 4",
        "contrast phase #4 - axial view": "post-contrast phase 4",
        "post-contrast phase 4": "post-contrast phase 4",
        "fifth post-contrast phase sequence": "post-contrast phase 5",
        "fifth postcontrast t1 fat saturated dynamics": "post-contrast phase 5",
        "registered ax sen vibrant multi phase - contrast phase: 5": "post-contrast phase 5",
        "contrast phase #5 - axial view": "post-contrast phase 5",
        "post-contrast phase 5": "post-contrast phase 5",
        "d0": "pre-contrast",
        "d1": "post-contrast phase 1",
        "d2": "post-contrast phase 2",
        "d3": "post-contrast phase 3",
        "d4": "post-contrast phase 4",
        "d5": "post-contrast phase 5",
        "t1": "t1",
        "t2": "t2",
        "adc": "adc",
        "dif": "diffusion",
        "f1": "post-contrast phase 1",
        "f2": "post-contrast phase 2",
        "f3": "post-contrast phase 3",
        "f4": "post-contrast phase 4",
        "f5": "post-contrast phase 5",
    }

    if x in contrast_phase_mapping:
        return contrast_phase_mapping[x]
    else:
        warn_unmapped("contrast phase", value)
        return None


# The projection an image was acquired in is exam context, not a BI-RADS finding
# descriptor, so it is not versioned with the lexicon.
# sources: https://dicom.nema.org/medical/dicom/2023c/output/chtml/part16/sect_CID_4015.html
_DVIEW_CODES = {
    "cc": CRANIAL_CAUDAL,
    "mlo": MEDIOLATERAL_OBLIQUE,
    "ml": "mediolateral",
    "lm": "lateromedial",
    "lmo": "lateromedial oblique",
    "at": "axillary tail",
    "cv": "cleavage",
    "fb": "from below",
    "rl": "rolled laterally",
    "rm": "rolled medially",
    "tan": "tangential",
    "sio": "superior inferior oblique",
    "iso": "inferior superior oblique",
    "xxcl": "laterally exaggerated cranial caudal",
    "xccm": "medially exaggerated cranial caudal",
    "xccl": "laterally exaggerated cranial caudal",  # typo in some reports
    "mloid": "mediolateral oblique implant displaced",
    "ccid": "craniocaudal implant displaced",
    "mlid": "mediolateral implant displaced",
    "lmid": "lateromedial implant displaced",
}

dview = {**_DVIEW_CODES, **{name: name for name in _DVIEW_CODES.values()}}


def get_proper_exam_view(value) -> str | None:
    if isna_v2(value):
        return None

    x = normalize_lexicon_value(value)

    valid_values = [MEDIOLATERAL_OBLIQUE, CRANIAL_CAUDAL]

    valid_typos = {
        "cc": CRANIAL_CAUDAL,
        "mlo": MEDIOLATERAL_OBLIQUE,
        # DICOM ViewPosition / ViewCodeSequence CodeMeaning spellings
        "cranio-caudal": CRANIAL_CAUDAL,
        "craniocaudal": CRANIAL_CAUDAL,
        "cranial-caudal": CRANIAL_CAUDAL,
        "medio-lateral oblique": MEDIOLATERAL_OBLIQUE,
        "mediolateral-oblique": MEDIOLATERAL_OBLIQUE,
        "medio-lateral-oblique": MEDIOLATERAL_OBLIQUE,
    }

    if x in valid_values:
        return x

    if x in valid_typos:
        return valid_typos[x]

    warn_unmapped("exam view", value, "outside the CC/MLO vocabulary this corpus keeps")
    return None


@trycatch_func
def column_cleaning_csv_reading(csv_path: str, delimiter: str = ",") -> pd.DataFrame:
    df = pd.read_csv(csv_path, delimiter=delimiter)
    df.columns = csv_column_cleaning(list(df.columns))
    return df


@trycatch_func
def csv_column_cleaning(cols: list[str]) -> list[str]:
    cleaned_cols = [
        upper_to_lower_wspace(
            col.strip().replace("_", " ").replace("(", " ").replace(")", " ")
        )
        for col in cols
    ]

    # remove any duplicate spaces that may have been introduced
    cleaned_cols = [re.sub(r"\s+", " ", col) for col in cleaned_cols]
    cleaned_cols = [col.strip() for col in cleaned_cols]

    return cleaned_cols


def lookup_key(value: Any) -> str:
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        text = str(int(value))
    elif isinstance(value, numbers.Real) and not isinstance(value, bool):
        as_float = float(value)
        text = str(int(as_float)) if as_float.is_integer() else str(as_float)
    else:
        text = str(value)

    return re.sub(r"\s+", " ", text).strip().lower()


def get_value(value: Any, map_set: dict) -> str | None:
    if value is None:
        return None
    return map_set.get(lookup_key(value), None)


def get_value_default(value: Any, map_set: dict) -> Any:
    if value is None:
        return None

    default_value = value.strip().lower() if isinstance(value, str) else value
    return map_set.get(lookup_key(value), default_value)


yes_no_mapping = {
    "0": "no",
    "1": "yes",
    "x": "yes",
    "y": "yes",
    "n": "no",
    "yes": "yes",
    "no": "no",
}

_RACE_SOURCE_SPELLINGS = {
    "caucasian or white": "white",
    "caucasian": "white",
    "african american": "black",
    "native american": "american indian",
    "african american or black": "black",
    "native hawaiian or other pacific islander": "native hawaiian",
    "american indian or alaskan native": "american indian",
    "black/african american": "black",
    "american indian/alaskan native": "american indian",
    "multiple races reported": "multiple",
    "amer indian": "american indian",
    "american indian": "american indian",
    "hawaian": "native hawaiian",
    "native hawaiian/pacific islander": "native hawaiian",
    "hawaiian/pacific islander": "native hawaiian",
    "multi": "multiple",
    "hispanic": "hispanic",
    "multiple race": "multiple",
    "hawa": "native hawaiian",
    "asian": "asian",
}

race_mappings = {
    **_RACE_SOURCE_SPELLINGS,
    **{name: name for name in _RACE_SOURCE_SPELLINGS.values()},
}


def sanitize_string(s: str) -> str:
    s = s.strip()
    s = s.replace("_", " ")
    s = s.replace("(", " ")
    s = s.replace(")", " ")
    s = re.sub(r"\s+", " ", s)  # replace multiple spaces with a single space
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
    s = s.lower()
    s = s.strip()

    # Case for strings that are joined by an "&", ",", "and", or "or"
    # Use only the first matching delimiter to avoid cross-contamination between
    # groups -- the recursive call below already handles any nested delimiters,
    # so continuing the loop would re-split the string we just rebuilt.
    delimiters = ["&", ",", " and ", " or "]
    for delim in delimiters:
        if delim in s:
            parts = [part.strip() for part in s.split(delim)]
            sanitized_parts = sorted([sanitize_string(part) for part in parts])
            s = f" {delim} ".join(sanitized_parts)
            break
    return s


@trycatch_func
@log_func_info
def draw_rectangle_on_image(
    image: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    out = np.array(image, copy=True)
    fill = (
        float(np.iinfo(out.dtype).max) if np.issubdtype(out.dtype, np.integer) else 1.0
    )
    cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), fill, -1)
    return out


def flatten_dict(d: str | dict | list, parent_key: str = "", sep: str = ".") -> dict:
    if isinstance(d, str):
        d = json.loads(d)

    flat_dict = {}
    if isinstance(d, dict):
        items = d.items()
    elif isinstance(d, list):
        items = enumerate(d)
    else:
        # Scalar value at top level — nothing to flatten
        return {parent_key: d} if parent_key else {}

    for key, value in items:
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, (dict, list)):
            flat_dict.update(flatten_dict(value, new_key, sep))
        else:
            flat_dict[new_key] = value
    return flat_dict


def get_or_default(value: Any, default: Any) -> Any:
    if isna_v2(value):
        return default
    return value


def isna_v2(value: Any) -> bool:
    na = pd.isna(value)
    if isinstance(na, (bool, np.bool_)):
        return bool(na)
    if hasattr(na, "__iter__"):
        return bool(np.all(np.asarray(na, dtype=bool)))
    return False


def normalize_lexicon_value(value) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _normalize_for_match(value) -> str:
    """Collapse every separator legacy exports use so surfaces match uniformly."""
    s = normalize_lexicon_value(value)
    s = s.replace("\u2019", "'").replace("\u02bc", "'")
    s = re.sub(r"[,/&_+\-\u2013\u2014]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _scan(value, synonyms: dict[str, str | None]) -> set[str]:
    """Find every known surface form in `value`, longest first, non-overlapping."""
    haystack = _normalize_for_match(value)
    found: set[str] = set()
    for surface in sorted(synonyms, key=len, reverse=True):
        needle = _normalize_for_match(surface)
        pattern = re.compile(rf"(?<![a-z]){re.escape(needle)}(?![a-z])")
        if pattern.search(haystack):
            target = synonyms[surface]
            if target is not None:
                found.add(target)
            haystack = pattern.sub(" ", haystack)
    return found


def _most_suspicious(found: Iterable[str], order: tuple[str, ...]) -> str | None:
    ranked = [t for t in order if t in set(found)]
    return ranked[-1] if ranked else None


def _check_modality(modality: str, allowed: tuple[str, ...], field: str) -> None:
    if modality not in allowed:
        raise ValueError(
            f"{field} is defined for {allowed} in BI-RADS {LEXICON_VERSION}; "
            f"got modality={modality!r}"
        )


# ===========================================================================
# BREAST TISSUE
# ===========================================================================

# --- Mammography: breast composition (Mammo p.173) -------------------------
BREAST_COMPOSITION_ORDER = (
    "(a) the breasts are almost entirely fatty",
    "(b) there are scattered areas of fibroglandular density",
    "(c) the breasts are heterogeneously dense, which may obscure small masses",
    "(d) the breasts are extremely dense, which lowers the sensitivity of mammography",
)

_A, _B, _C, _D = BREAST_COMPOSITION_ORDER

_BREAST_COMPOSITION_SYNONYMS = {
    "a": _A,
    "1": _A,
    "fatty": _A,
    "almost entirely fatty": _A,
    "the breasts are almost entirely fat": _A,
    "the breasts are almost entirely fatty": _A,
    "b": _B,
    "2": _B,
    "scattered": _B,
    "scattered fibroglandular densities": _B,
    "scattered areas of fibroglandular density": _B,
    "c": _C,
    "3": _C,
    "heterogeneously dense": _C,
    "heterogeneous dense": _C,
    "d": _D,
    "4": _D,
    "extremely dense": _D,
    # not breast-composition assessments
    "normal": None,
    "normal male": None,
    "0": None,
}


def get_proper_breast_density(value) -> str | None:
    """Mammographic breast composition (a)-(d). Mammography only."""
    if isna_v2(value):
        return None
    x = normalize_lexicon_value(value)
    if x in BREAST_COMPOSITION_ORDER:
        return x
    # lookup_key, not x: sheets store the ACR code as a number, so it reaches
    # here as "1.0" rather than "1".
    key = lookup_key(value)
    if key in _BREAST_COMPOSITION_SYNONYMS:
        return _BREAST_COMPOSITION_SYNONYMS[key]
    warn_unmapped("breast composition", value)
    return None


# --- MRI: amount of fibroglandular tissue (MRI p.717) ----------------------
MRI_FGT_ORDER = (
    "(a) almost entirely fatty",
    "(b) scattered fibroglandular tissue",
    "(c) heterogeneous fibroglandular tissue",
    "(d) extreme fibroglandular tissue",
)

_FA, _FB, _FC, _FD = MRI_FGT_ORDER

_MRI_FGT_SYNONYMS = {
    "a": _FA,
    "1": _FA,
    "almost entirely fatty": _FA,
    "fatty": _FA,
    "b": _FB,
    "2": _FB,
    "scattered fibroglandular tissue": _FB,
    "scattered": _FB,
    "c": _FC,
    "3": _FC,
    "heterogeneous fibroglandular tissue": _FC,
    "heterogeneous": _FC,
    "d": _FD,
    "4": _FD,
    "extreme fibroglandular tissue": _FD,
    "extreme": _FD,
    "extremely dense": _FD,
}


def get_proper_mri_fgt(value) -> str | None:
    """MRI amount of fibroglandular tissue (FGT)."""
    if isna_v2(value):
        return None
    x = normalize_lexicon_value(value)
    if x in MRI_FGT_ORDER:
        return x
    key = lookup_key(value)  # see get_proper_breast_density
    if key in _MRI_FGT_SYNONYMS:
        return _MRI_FGT_SYNONYMS[key]
    warn_unmapped("mri fgt", value)
    return None


# --- Ultrasound: tissue composition (US p.488) -----------------------------
TISSUE_COMPOSITION_VALUES = (
    "homogeneous background echotexture - fat",
    "homogeneous background echotexture - fibroglandular",
    "heterogeneous background echotexture",
)

_TISSUE_COMPOSITION_SYNONYMS = {
    "heterogeneous": "heterogeneous background echotexture",
    "heterogeneous background echotexture": "heterogeneous background echotexture",
    "heterogeneous: predominantly fat": "heterogeneous background echotexture",
    "heterogeneous: predominantly fibroglandular": "heterogeneous background echotexture",
    "lactating&heterogeneous: predominantly fibroglandular": "heterogeneous background echotexture",
    "lactating&heterogeneous: predominantly fat": "heterogeneous background echotexture",
    "homogeneous: fibroglandular": "homogeneous background echotexture - fibroglandular",
    "lactating&homogeneous: fibroglandular": "homogeneous background echotexture - fibroglandular",
    "homogeneous background echotexture fibroglandular": "homogeneous background echotexture - fibroglandular",
    "homogeneous: fat": "homogeneous background echotexture - fat",
    "homogeneous background echotexture fat": "homogeneous background echotexture - fat",
    "lactating": None,
}


def get_proper_tissue_composition(value) -> str | None:
    """Ultrasound background echotexture."""
    if isna_v2(value):
        return None
    x = normalize_lexicon_value(value)
    if x in TISSUE_COMPOSITION_VALUES:
        return x
    if x in _TISSUE_COMPOSITION_SYNONYMS:
        return _TISSUE_COMPOSITION_SYNONYMS[x]
    y = _normalize_for_match(value)
    if y in {_normalize_for_match(v) for v in TISSUE_COMPOSITION_VALUES}:
        for v in TISSUE_COMPOSITION_VALUES:
            if _normalize_for_match(v) == y:
                return v
    warn_unmapped("tissue composition", value)
    return None


# ===========================================================================
# MASSES -- SHAPE (identical vocabulary across mammography, US and MRI)
# ===========================================================================
# Mammo p.173 / US p.488 / MRI p.717 all read: Oval, Lobulated, Round, Irregular.
MASS_SHAPE_ORDER = ("oval", "lobulated", "round", "irregular")

_MASS_SHAPE_SYNONYMS: dict[str, str | None] = {
    "oval": "oval",
    "ovoid": "oval",
    "elliptical": "oval",
    "egg shaped": "oval",
    "lobulated": "lobulated",
    "lobular": "lobulated",
    "macrolobulated": "lobulated",
    "round": "round",
    "rounded": "round",
    "spherical": "round",
    "ball shaped": "round",
    "circular": "round",
    "irregular": "irregular",
    "polygonal": "irregular",
}

# Findings that legitimately are not mass shapes
_NON_SHAPE_SURFACES = {
    "architectural distortion",
    "questioned architectural distortion",
    "asymmetry",
    "global asymmetry",
    "focal asymmetry",
    "focal asymmetric density",
    "focal asymetry",
    "developing asymmetry",
    "developing asymmestry",
    "asymmetric breast tissue",
    "lymph node",
    "lumph node",
    "intramammary lymph node",
    "axillary lymph node",
    "non mass lesion",
    "non mass enhancement",
    "asymmetric tubular structure",
    "asymmectric tubular structure",
    "solitary dilated duct",
    "skin lesion",
    "generic",
    "dilated duct",
    "multiple dilated ducts",
    "nme"
}


def get_mass_shape_tokens(value) -> set[str]:
    """Every v2025 shape term present in a (possibly compound) legacy value."""
    if isna_v2(value):
        return set()
    return _scan(value, _MASS_SHAPE_SYNONYMS)


def get_proper_mass_shape(value) -> str | None:
    """Dominant mass shape. Compound values resolve to the most suspicious term."""
    if isna_v2(value):
        return None

    tokens = get_mass_shape_tokens(value)
    if tokens:
        return _most_suspicious(tokens, MASS_SHAPE_ORDER)

    if _scan(value, {s: s for s in _NON_SHAPE_SURFACES}):
        return None  # a real finding, just not a mass shape

    warn_unmapped("mass shape", value)
    return None


# ===========================================================================
# MASSES -- MARGIN (modality-specific in v2025)
# ===========================================================================
# Mammography (p.173): Circumscribed, Obscured, Indistinct, Spiculated.
# Ultrasound (p.488-489): Circumscribed | Non-circumscribed -> Indistinct,
# MRI (p.717): Circumscribed | Non-circumscribed -> Indistinct, Spiculated.

MARGIN_ORDER = {
    MAMMOGRAPHY: ("circumscribed", "obscured", "indistinct", "spiculated"),
    ULTRASOUND: (
        "circumscribed",
        "non-circumscribed",
        "indistinct",
        "angular",
        "microlobulated",
        "spiculated",
    ),
    MRI: ("circumscribed", "non-circumscribed", "indistinct", "spiculated"),
}

_MARGIN_SYNONYMS_COMMON: dict[str, str | None] = {
    "circumscribed": "circumscribed",
    "well defined": "circumscribed",
    "well circumscribed": "circumscribed",
    "sharply defined": "circumscribed",
    "indistinct": "indistinct",
    "ill defined": "indistinct",
    "ill-defined": "indistinct",
    "poorly defined": "indistinct",
    "spiculated": "spiculated",
    "speculated": "spiculated",
    "stellate": "spiculated",
}

_MARGIN_SYNONYMS = {
    MAMMOGRAPHY: {
        **_MARGIN_SYNONYMS_COMMON,
        "obscured": "obscured",
        "partially obscured": "obscured",
    },
    ULTRASOUND: {
        **_MARGIN_SYNONYMS_COMMON,
        "non circumscribed": "non-circumscribed",
        "not circumscribed": "non-circumscribed",
        "angular": "angular",
        "microlobulated": "microlobulated",
    },
    MRI: {
        **_MARGIN_SYNONYMS_COMMON,
        "non circumscribed": "non-circumscribed",
        "not circumscribed": "non-circumscribed",
        "smooth": "circumscribed",
        "irregular": "indistinct",
    },
}

_MICROLOBULATED_RE = re.compile(r"(?<![a-z])microlobulated(?![a-z])")


def get_proper_mass_margin(value, modality: str) -> str | None:
    """Mass margin. `modality` is required -- the vocabulary differs in v2025."""
    _check_modality(modality, MODALITIES, "mass margin")
    if isna_v2(value):
        return None

    tokens = _scan(value, _MARGIN_SYNONYMS[modality])

    # A bare parent term is only meaningful if no leaf was also present.
    if len(tokens) > 1:
        tokens.discard("non-circumscribed")

    if tokens:
        return _most_suspicious(tokens, MARGIN_ORDER[modality])

    if modality == MAMMOGRAPHY and _MICROLOBULATED_RE.search(
        _normalize_for_match(value)
    ):
        warn_unmapped(
            "mass margin",
            value,
            "'microlobulated' was removed from the v2025 mammography margin list; "
            "run migrate_mammography_microlobulated() to rewrite it as 'indistinct'",
        )
        return None

    if modality in (ULTRASOUND, MRI) and _scan(value, {"obscured": "obscured"}):
        warn_unmapped(
            "mass margin",
            value,
            f"'obscured' is a mammography-only margin; no {modality} equivalent",
        )
        return None

    warn_unmapped(f"mass margin ({modality})", value)
    return None


def migrate_mammography_microlobulated(
    shape_value, margin_value
) -> tuple[str | None, str | None]:
    """Map a legacy mammography (shape, margin) pair onto the v2025 vocabulary.

    v2025 retired "microlobulated" from the mammography margin list and directs
    that such a margin be reported as INDISTINCT. It is deliberately not folded
    into the "lobulated" shape that v2025 restored: microlobulated carries a
    suspicious connotation, whereas a lobulated shape does not, so moving it
    there would launder a suspicious descriptor into a neutral one.

    (Ultrasound keeps "microlobulated" as a margin, so this migration is
    mammography-only.)
    """
    shape = get_proper_mass_shape(shape_value)

    if margin_value is None or isna_v2(margin_value):
        return shape, None

    normalized = _normalize_for_match(margin_value)
    if not _MICROLOBULATED_RE.search(normalized):
        return shape, get_proper_mass_margin(margin_value, MAMMOGRAPHY)

    # Rewrite in place rather than stripping the token, so a compound value such
    # as "microlobulated, spiculated" still resolves to the more suspicious term
    # through the normal precedence rules instead of losing the residue.
    rewritten = _MICROLOBULATED_RE.sub("indistinct", normalized)
    return shape, get_proper_mass_margin(rewritten, MAMMOGRAPHY)


# ===========================================================================
# MASSES -- DENSITY (mammography only, Mammo p.173)
# ===========================================================================
MASS_DENSITY_ORDER = ("fat-containing", "low density", "equal density", "high density")

_MASS_DENSITY_SYNONYMS: dict[str, str | None] = {
    "fat containing": "fat-containing",
    "fat-containing": "fat-containing",
    "radiolucent": "fat-containing",
    "mixed": "fat-containing",  # Manual: fat-containing includes mixed-density lesions
    "low": "low density",
    "low density": "low density",
    "equal": "equal density",
    "equal density": "equal density",
    "isodense": "equal density",
    "high": "high density",
    "high density": "high density",
}


def get_proper_mass_density(value) -> str | None:
    if isna_v2(value):
        return None
    tokens = _scan(value, _MASS_DENSITY_SYNONYMS)
    if tokens:
        return _most_suspicious(tokens, MASS_DENSITY_ORDER)
    warn_unmapped("mass density", value)
    return None


# ===========================================================================
# MASSES -- ULTRASOUND ECHO PATTERN, POSTERIOR FEATURES
# ===========================================================================
# --- Mass echo pattern (US p.489) ------------------------------------------
# "complex cystic and solid" no longer appears in the v2025 mass echo
# pattern list; "mixed solid and cystic" replaces it.
MASS_ECHO_PATTERN_ORDER = (
    "anechoic",
    "hyperechoic",
    "isoechoic",
    "hypoechoic",
    "heterogeneous",
    "mixed solid and cystic",
)

_MASS_ECHO_SYNONYMS: dict[str, str | None] = {
    "anechoic": "anechoic",
    "noecho": "anechoic",
    "no echo": "anechoic",
    "hyperechoic": "hyperechoic",
    "highecho": "hyperechoic",
    "isoechoic": "isoechoic",
    "hypoechoic": "hypoechoic",
    "lowecho": "hypoechoic",
    "slightlylowecho": "hypoechoic",
    "heterogeneous": "heterogeneous",
    "heteroechoic": "heterogeneous",
    "heterogeneousecho": "heterogeneous",
    "mixed solid and cystic": "mixed solid and cystic",
    "complex cystic and solid": "mixed solid and cystic",
    "complex cystic solid": "mixed solid and cystic",
    "cysticsolidmixedecho": "mixed solid and cystic",
    "solid and cystic": "mixed solid and cystic",
}


def get_proper_mass_echo_pattern(value) -> str | None:
    """Ultrasound mass echo pattern. (Was get_proper_mass_echotexture.)"""
    if isna_v2(value):
        return None
    tokens = _scan(value, _MASS_ECHO_SYNONYMS)
    if tokens:
        return _most_suspicious(tokens, MASS_ECHO_PATTERN_ORDER)
    warn_unmapped("mass echo pattern", value)
    return None


# --- Posterior features (US p.489) -----------------------------------------
# "combined pattern" was dropped in v2025. Only three options remain,
# and they apply to both masses and non-mass lesions.
POSTERIOR_FEATURES_ORDER = ("no posterior features", "enhancement", "shadowing")

_POSTERIOR_SYNONYMS: dict[str, str | None] = {
    "no posterior features": "no posterior features",
    "none": "no posterior features",
    "no": "no posterior features",
    "absent": "no posterior features",
    "enhancement": "enhancement",
    "posterior enhancement": "enhancement",
    "shadowing": "shadowing",
    "posterior shadowing": "shadowing",
    "combined pattern": None,
    "combined": None,
    "mixed": None,
    "present": None,
}


def get_proper_posterior_features(value) -> str | None:
    if isna_v2(value):
        return None
    x = normalize_lexicon_value(value)
    if x in POSTERIOR_FEATURES_ORDER:
        return x
    if x in _POSTERIOR_SYNONYMS:
        return _POSTERIOR_SYNONYMS[x]
    warn_unmapped("posterior features", value)
    return None


# ===========================================================================
# CALCIFICATIONS (Mammo pp.173-174)
# ===========================================================================
# Typically benign (v2025): skin, vascular, coarse, large rod-like, round, rim,
#   layering, suture.
#   - "milk of calcium" -> "layering"
#   - "punctate"        -> folded into "round"
#   - "dystrophic"      -> folded into "coarse"
#   - "popcorn-like"    -> folded into "coarse"
# Suspicious morphology (v2025, unchanged): amorphous, coarse heterogeneous,
#   fine pleomorphic, fine linear or fine linear-branching.

SUSPICIOUS_CALCIFICATIONS = (
    "amorphous",
    "coarse heterogeneous",
    "fine pleomorphic",
    "fine linear or fine linear-branching",
)

TYPICALLY_BENIGN_CALCIFICATIONS = (
    "skin",
    "vascular",
    "coarse",
    "large rod-like",
    "round",
    "rim",
    "layering",
    "suture",
)

# Ascending order of concern; the last match wins.
SUSPICIOUS_PRECEDENCE = SUSPICIOUS_CALCIFICATIONS

# Nothing benign is "more concerning", so this ranks by how specific/diagnostic
# the morphology is. Ascending preference.
BENIGN_PRECEDENCE = (
    "round",
    "coarse",
    "rim",
    "layering",
    "large rod-like",
    "skin",
    "vascular",
    "suture",
)

CALCIFICATION_SYNONYMS: dict[str, str] = {
    # -- suspicious morphology ---------------------------------------------
    "fine linear or fine linear-branching": "fine linear or fine linear-branching",
    "fine linear or fine linear branching": "fine linear or fine linear-branching",
    "fine linear branching (casting)": "fine linear or fine linear-branching",
    "fine linear-branching (casting)": "fine linear or fine linear-branching",
    "fine-linear (casting)": "fine linear or fine linear-branching",
    "fine and linear calcification": "fine linear or fine linear-branching",
    "fine linear branching": "fine linear or fine linear-branching",
    "fine linear-branching": "fine linear or fine linear-branching",
    "fine and linear": "fine linear or fine linear-branching",
    "fine linear": "fine linear or fine linear-branching",
    "casting": "fine linear or fine linear-branching",
    "fine pleomorphic": "fine pleomorphic",
    "pleomorphic": "fine pleomorphic",
    "coarse heterogeneous": "coarse heterogeneous",
    "amorphous and indistinct": "amorphous",
    "amorphous": "amorphous",
    # -- typically benign ---------------------------------------------------
    "large rod-like": "large rod-like",
    "large rod like": "large rod-like",
    "large rodlike": "large rod-like",
    "rod like": "large rod-like",
    # v2025: milk of calcium -> layering
    "layering": "layering",
    "milk of calcium": "layering",
    "sedimented": "layering",
    # v2025: dystrophic and popcorn-like both fold into coarse
    "coarse popcorn-like": "coarse",
    "course popcorn-like": "coarse",
    "popcorn-like": "coarse",
    "popcorn": "coarse",
    "dystrophic": "coarse",
    "coarse": "coarse",
    "vascular": "vascular",
    "suture": "suture",
    "skin": "skin",
    # v2025: punctate folded into round
    "round and regular": "round",
    "small round calcification": "round",
    "punctate": "round",
    "round": "round",
    # rim (eggshell / lucent-centred consolidated in the 5th ed)
    "lucent centered": "rim",
    "lucent center": "rim",
    "lucent-centered": "rim",
    "eggshell": "rim",
    "egg shell": "rim",
    "oil cyst": "rim",
    "rim": "rim",
}

CALCIFICATION_FAMILY_ONLY_TOKENS = {"benign": "typically benign"}
CALCIFICATION_NON_MORPHOLOGY_TOKENS = {
    "benign",
    "generic",
    "calcification",
    "calc",
    "calcifications",
    "microcalcifications",
}


def find_calcification_tokens(text) -> set[str]:
    return _scan(text, CALCIFICATION_SYNONYMS)


def get_proper_calcification_type(value) -> tuple[str | None, str | None]:
    if isna_v2(value):
        return None, None

    tokens = find_calcification_tokens(value)

    suspicious = _most_suspicious(tokens, SUSPICIOUS_PRECEDENCE)
    if suspicious:
        return "suspicious morphology", suspicious

    benign = _most_suspicious(tokens, BENIGN_PRECEDENCE)
    if benign:
        return "typically benign", benign

    residue = set(_normalize_for_match(value).split())
    if residue and residue <= CALCIFICATION_NON_MORPHOLOGY_TOKENS:
        for token, family in CALCIFICATION_FAMILY_ONLY_TOKENS.items():
            if token in residue:
                return family, None
        return None, None

    warn_unmapped("calcification type", value)
    return None, None


# --- Distribution (Mammo p.174) -- vocabulary unchanged in v2025 -----------
CALCIFICATION_DISTRIBUTION_ORDER = (
    "diffuse",
    "regional",
    "grouped",
    "linear",
    "segmental",
)

_CALC_DISTRIBUTION_SYNONYMS: dict[str, str | None] = {
    "diffuse": "diffuse",
    "diffusely scattered": "diffuse",
    "scattered": "diffuse",
    "regional": "regional",
    "grouped": "grouped",
    "clustered": "grouped",
    "linear": "linear",
    "segmental": "segmental",
}


def get_proper_calcification_distribution(value) -> str | None:
    if isna_v2(value):
        return None
    tokens = _scan(value, _CALC_DISTRIBUTION_SYNONYMS)
    if tokens:
        return _most_suspicious(tokens, CALCIFICATION_DISTRIBUTION_ORDER)
    warn_unmapped("calcification distribution", value)
    return None


# ===========================================================================
# FINDING TYPE (the single vocabulary for "what kind of finding is this")
# ===========================================================================
# One canonical spelling per finding category, used by BOTH the routing mapper
# and the adapters that build a finding directly. Previously the two disagreed
# ("calcification" when built directly, "calcifications" when routed), which put
# two spellings of the same class into the VQA answer space and made the
# lesion-type applicability rules miss half the rows.
MASS = "mass"
CALCIFICATION = "calcification"
NON_MASS_LESION = "non-mass lesion"
NON_MASS_ENHANCEMENT = "non-mass enhancement"
ARCHITECTURAL_DISTORTION = "architectural distortion"
ASYMMETRY = "asymmetry"
LYMPH_NODE = "lymph node"
SKIN_LESION = "skin lesion"
DILATED_DUCT = "dilated duct"

FINDING_TYPES = {
    MAMMOGRAPHY: (
        MASS,
        CALCIFICATION,
        ARCHITECTURAL_DISTORTION,
        ASYMMETRY,
        LYMPH_NODE,
        SKIN_LESION,
        DILATED_DUCT,
    ),
    ULTRASOUND: (
        MASS,
        NON_MASS_LESION,
        CALCIFICATION,
        LYMPH_NODE,
    ),
    MRI: (
        MASS,
        NON_MASS_ENHANCEMENT,
        LYMPH_NODE,
    ),
}

# Every finding type any modality can carry, in one flat tuple, so downstream
# code can validate a value without knowing the modality.
ALL_FINDING_TYPES = tuple(
    dict.fromkeys(t for types in FINDING_TYPES.values() for t in types)
)

# Which detail object a finding type is allowed to carry. A finding type absent
# from this mapping is a bare categorical finding: it is still recorded, located
# and counted, it just has no descriptor block (architectural distortion,
# asymmetry, non-mass lesion/enhancement, lymph node, skin lesion, dilated duct).
#
# Keys must match the `*_details` fields `Lesion` actually defines. Naming a
# payload here that the dataclass does not have makes every finding of that type
# unconstructible, so the two are asserted to agree at import time by
# dtos.json_report_findings.
FINDING_TYPE_DETAILS = {
    MASS: "mass_details",
    CALCIFICATION: "calcification_details",
}

# Most specific first. Used only to break ties when one legacy value names
# several categories at once.
_FINDING_TYPE_PRECEDENCE = (
    CALCIFICATION,
    NON_MASS_ENHANCEMENT,
    NON_MASS_LESION,
    ARCHITECTURAL_DISTORTION,
    ASYMMETRY,
    DILATED_DUCT,
    LYMPH_NODE,
    SKIN_LESION,
    MASS,
)

_FINDING_TYPE_SYNONYMS: dict[str, str] = {
    "mass": MASS,
    "nodule": MASS,
    "tumor": MASS,
    "non mass lesion": NON_MASS_LESION,
    "non-mass lesion": NON_MASS_LESION,
    "nonmass": NON_MASS_LESION,
    "non mass enhancement": NON_MASS_ENHANCEMENT,
    "nme": NON_MASS_ENHANCEMENT,
    "calcification": CALCIFICATION,
    "calcifications": CALCIFICATION,
    "calc": CALCIFICATION,
    "microcalcifications": CALCIFICATION,
    "architectural distortion": ARCHITECTURAL_DISTORTION,
    "questioned architectural distortion": ARCHITECTURAL_DISTORTION,
    "asymmetry": ASYMMETRY,
    "global asymmetry": ASYMMETRY,
    "focal asymmetry": ASYMMETRY,
    "focal asymmetric density": ASYMMETRY,
    "asymmetric breast tissue": ASYMMETRY,
    "developing asymmetry": ASYMMETRY,
    # misspellings that the EMBED legend ships verbatim
    "focal asymetry": ASYMMETRY,
    "developing asymmestry": ASYMMETRY,
    "lymph node": LYMPH_NODE,
    "lumph node": LYMPH_NODE,
    "intramammary lymph node": LYMPH_NODE,
    "axillary lymph node": LYMPH_NODE,
    "skin lesion": SKIN_LESION,
    "solitary dilated duct": DILATED_DUCT,
    "multiple dilated ducts": DILATED_DUCT,
    "dilated duct": DILATED_DUCT,
    # BI-RADS treats an asymmetric tubular structure and a solitary dilated duct
    # as the same finding; EMBED spells the first one two ways.
    "asymmetric tubular structure": DILATED_DUCT,
    "asymmectric tubular structure": DILATED_DUCT,
}


def get_proper_finding_type(value, modality: str) -> str | None:
    """
    Route a legacy label to a v2025 finding category.

    Call this BEFORE the mass mappers: a row whose 'mass shape' column reads
    'architectural_distortion' is not a mass with an unmappable shape, it is an
    architectural distortion, and on ultrasound it may now be a non-mass lesion.
    """
    _check_modality(modality, MODALITIES, "finding type")
    if isna_v2(value):
        return None

    tokens = _scan(value, _FINDING_TYPE_SYNONYMS)
    allowed = FINDING_TYPES[modality]
    tokens = {t for t in tokens if t in allowed}
    if len(tokens) == 1:
        return tokens.pop()
    if tokens:
        # A compound value names several categories. Resolve it by a fixed
        # specificity order rather than alphabetically, so the choice is
        # reproducible and explainable: the descriptor-bearing categories win
        # over the bare categorical ones, and the generic "mass" loses to
        # anything more specific.
        for candidate in _FINDING_TYPE_PRECEDENCE:
            if candidate in tokens:
                return candidate
        return sorted(tokens)[0]

    warn_unmapped(f"finding type ({modality})", value)
    return None


# ===========================================================================
# ULTRASOUND ASSOCIATED FEATURES (US p.490)
# ===========================================================================
# v2025 distinguishes two echogenic boundary features:
#   echogenic pseudocapsule -- THIN, uniform, around an otherwise circumscribed
#                              margin (benign connotation)
#   echogenic rind          -- THICK band that disrupts surrounding texture
#                              (associated with invasive cancer)
ECHOGENIC_FEATURES = ("echogenic pseudocapsule", "echogenic rind")

_ECHOGENIC_SYNONYMS: dict[str, str | None] = {
    "echogenic pseudocapsule": "echogenic pseudocapsule",
    "pseudocapsule": "echogenic pseudocapsule",
    "thin echogenic capsule": "echogenic pseudocapsule",
    "echogenic rind": "echogenic rind",
    "rind": "echogenic rind",
    "thick echogenic band": "echogenic rind",
    "echogenic halo": None,
    # A bare halo flag -- including the yes/no columns legacy sheets ship -- says
    # an echogenic boundary is there but not whether it is thin or thick, which
    # is exactly the distinction v2025 introduced. It cannot be resolved.
    "halo": None,
    "yes": None,
    "no": None,
    "present": None,
    "absent": None,
}

# The 5th edition's "halo" associated feature was REMOVED in v2025 and replaced
# by two distinct features: echogenic pseudocapsule (a thin, uniform boundary
# with a benign connotation) and echogenic rind (a thick band that disrupts the
# surrounding texture, associated with invasive cancer). A source column that
# only records halo yes/no does not distinguish them, so it has no v2025
# representation and is dropped rather than carried under a retired term.


def get_proper_echogenic_feature(value) -> str | None:
    """Ultrasound echogenic boundary feature (replaces the 5th edition's halo)."""
    if isna_v2(value):
        return None
    tokens = _scan(value, _ECHOGENIC_SYNONYMS)
    if len(tokens) == 1:
        return tokens.pop()
    if tokens:
        return "echogenic rind"
    warn_unmapped(
        "echogenic feature",
        value,
        "v2025 splits this into echogenic pseudocapsule (thin) vs echogenic "
        "rind (thick); a bare halo flag cannot be resolved to either, so the "
        "annotation is dropped",
    )
    return None


# ===========================================================================
# ASSESSMENT CATEGORIES (Mammo p.176 / US p.491-492 / MRI p.720)
# ===========================================================================
# Category 0 is now TWO distinct statements. 4A/4B/4C appear as subdivisions on
# the mammography form only; the US and MRI forms list Category 4 undivided. The
# subdivisions are also inconsistently populated across sources, so every 4A/4B/4C
# is folded into an undivided Category 4 -- the assessment space is the same for
# all modalities.
BIRADS_INCOMPLETE_ADDITIONAL = "(0) incomplete - need additional imaging evaluation"
BIRADS_INCOMPLETE_PRIORS = "(0) incomplete - need prior imaging for comparison"

BIRADS_CATEGORIES = (
    BIRADS_INCOMPLETE_ADDITIONAL,
    BIRADS_INCOMPLETE_PRIORS,
    "(1) negative",
    "(2) benign",
    "(3) probably benign",
    "(4) suspicious",
    "(5) highly suggestive of malignancy",
    "(6) known biopsy-proven malignancy",
)

BIRADS_MAPPING = {
    0: BIRADS_INCOMPLETE_ADDITIONAL,
    1: "(1) negative",
    2: "(2) benign",
    3: "(3) probably benign",
    4: "(4) suspicious",
    5: "(5) highly suggestive of malignancy",
    6: "(6) known biopsy-proven malignancy",
}

_BIRADS_SYNONYMS: dict[str, str] = {
    "0": BIRADS_INCOMPLETE_ADDITIONAL,
    "incomplete": BIRADS_INCOMPLETE_ADDITIONAL,
    "additional evaluation": BIRADS_INCOMPLETE_ADDITIONAL,
    "need additional imaging evaluation": BIRADS_INCOMPLETE_ADDITIONAL,
    "incomplete: need additional imaging evaluation": BIRADS_INCOMPLETE_ADDITIONAL,
    "recall": BIRADS_INCOMPLETE_ADDITIONAL,
    "need prior mammograms for comparison": BIRADS_INCOMPLETE_PRIORS,
    "need prior imaging for comparison": BIRADS_INCOMPLETE_PRIORS,
    "incomplete: need prior imaging for comparison": BIRADS_INCOMPLETE_PRIORS,
    "incomplete: need prior mammograms for comparison": BIRADS_INCOMPLETE_PRIORS,
    "1": "(1) negative",
    "negative": "(1) negative",
    "2": "(2) benign",
    "benign": "(2) benign",
    "3": "(3) probably benign",
    "probably benign": "(3) probably benign",
    "4": "(4) suspicious",
    "suspicious": "(4) suspicious",
    "suspicious abnormality": "(4) suspicious",
    # The 4A/4B/4C subdivisions are folded into the undivided Category 4.
    "4a": "(4) suspicious",
    "4b": "(4) suspicious",
    "4c": "(4) suspicious",
    "(4a) suspicious - low suspicion for malignancy": "(4) suspicious",
    "(4b) suspicious - moderate suspicion for malignancy": "(4) suspicious",
    "(4c) suspicious - high suspicion for malignancy": "(4) suspicious",
    "low suspicion for malignancy": "(4) suspicious",
    "moderate suspicion for malignancy": "(4) suspicious",
    "high suspicion for malignancy": "(4) suspicious",
    "3, 4a": "(4) suspicious",
    "5": "(5) highly suggestive of malignancy",
    "highly suggestive of malignancy": "(5) highly suggestive of malignancy",
    "6": "(6) known biopsy-proven malignancy",
    "known biopsy proven": "(6) known biopsy-proven malignancy",
    "known biopsy-proven malignancy": "(6) known biopsy-proven malignancy",
}


def get_proper_birads(value, modality: str = MAMMOGRAPHY) -> str:
    """`modality` is accepted for symmetry with the other lexicon helpers; the
    assessment space no longer differs by modality now that 4A/4B/4C collapse."""
    if isna_v2(value):
        raise ValueError("BI-RADS value cannot be None or NaN")

    x = normalize_lexicon_value(value)
    x = re.sub(r"^bi[\s\-]?rads[\s:]*", "", x).strip()

    mapped = x if x in BIRADS_CATEGORIES else _BIRADS_SYNONYMS.get(x)
    if mapped is None:
        # lookup_key, not x: a numeric column that holds a NaN anywhere is read as
        # float, so category 2 arrives as "2.0" rather than "2".
        mapped = _BIRADS_SYNONYMS.get(lookup_key(value))
    if mapped is None:
        raise ValueError(f"Value {value} not recognized for BI-RADS {LEXICON_VERSION}")

    return mapped


# Category 6 is a *known* biopsy-proven malignancy: it is a statement about the
# patient's pathology record, not something a reader can derive from the pixels.
# A model asked to predict it from an image alone can only be learning a source
# artefact, so every adapter folds it into 5. This is enforced centrally rather
# than left to each adapter to remember -- previously some called
# collapse_known_biopsy_proven(), EMBED rewrote the code upstream, and the rest
# let category 6 through.
KNOWN_BIOPSY_PROVEN = "(6) known biopsy-proven malignancy"
HIGHLY_SUGGESTIVE = "(5) highly suggestive of malignancy"


def collapse_known_biopsy_proven(value):
    if value is None:
        return None
    if value == KNOWN_BIOPSY_PROVEN:
        return HIGHLY_SUGGESTIVE
    return value


def map_birads(value: Any, source: str = "", modality: str = MAMMOGRAPHY) -> str | None:
    """Normalise a source BI-RADS value, applying the project-wide policies.

    Returns None for category 0 and logs unrecognised values so a caller can
    drop the row rather than inventing an assessment. Category 6 folds into 5.
    """
    if isna_v2(value) or value is None:
        return None
    try:
        mapped = get_proper_birads(value, modality=modality)
    except ValueError:
        logger.warning(
            f"{source or 'BI-RADS'}: value '{value}' is not a recognized "
            f"{LEXICON_VERSION} assessment; the row will be dropped."
        )
        return None

    if birads_category_number(mapped) == 0:
        return None
    return collapse_known_biopsy_proven(mapped)


def birads_category_number(assessment: str | None) -> int | None:
    if assessment is None or isna_v2(assessment):
        return None
    match = re.match(r"^\((\d)[a-c]?\)", str(assessment))
    return int(match.group(1)) if match else None


# ===========================================================================
# LOCATION OF FINDING (Mammo p.176 / US p.491 / MRI p.719)
# ===========================================================================
# v2025 states clock-face position is PREFERRED, with quadrant as the
# alternative, plus retroareolar / central / axillary tail, plus a separate
# depth field. Collapsing a clock position to a quadrant therefore discards the
# preferred representation -- keep both.

CLOCK_RE = re.compile(
    r"^(?P<hour>1[0-2]|[1-9])\s*(?:o\s*'?\s*clock|:00|h)?(?:\s+position)?$"
)

QUADRANTS = ("upper outer", "upper inner", "lower outer", "lower inner")
SPECIAL_LOCATIONS = ("retroareolar", "central", "axillary tail")
DEPTHS = ("anterior third", "middle third", "posterior third")

_DEPTH_SYNONYMS = {
    "anterior": "anterior third",
    "anterior third": "anterior third",
    "middle": "middle third",
    "middle third": "middle third",
    "mid": "middle third",
    "posterior": "posterior third",
    "posterior third": "posterior third",
}

_QUADRANT_SYNONYMS: dict[str, str] = {
    "upper outer": "upper outer",
    "uoq": "upper outer",
    "upper inner": "upper inner",
    "uiq": "upper inner",
    "lower outer": "lower outer",
    "loq": "lower outer",
    "lower inner": "lower inner",
    "liq": "lower inner",
    "upper medial": "upper inner",
    "upper lateral": "upper outer",
    "lower medial": "lower inner",
    "lower lateral": "lower outer",
}

_SPECIAL_SYNONYMS: dict[str, str] = {
    "retroareolar": "retroareolar",
    "subareolar": "retroareolar",
    "sub areolar": "retroareolar",
    "central": "central",
    "axillary tail": "axillary tail",
    "tail of spence": "axillary tail",
}

# Cardinal directions are not v2025 summary-form terms. They are kept so legacy
# rows are not silently lost, but they are reported in `coarse_location`, not
# in `quadrant`, so downstream code cannot mistake them for a quadrant.
_CARDINAL_SYNONYMS: dict[str, str] = {
    "upper": "upper",
    "superior": "upper",
    "lower": "lower",
    "inferior": "lower",
    "medial": "medial",
    "inner": "medial",
    "lateral": "lateral",
    "outer": "lateral",
}

LATERALITY_MAP = {
    "r": "right",
    "right": "right",
    "right breast": "right",
    "l": "left",
    "left": "left",
    "left breast": "left",
    "b": "bilateral",
    "both": "bilateral",
    "bilateral": "bilateral",
    "both breasts": "bilateral",
}

# Right breast, patient perspective. Left is the mirror image.
CLOCK_TO_QUADRANT = {
    12: "upper",
    1: "upper inner",
    2: "upper inner",
    3: "medial",
    4: "lower inner",
    5: "lower inner",
    6: "lower",
    7: "lower outer",
    8: "lower outer",
    9: "lateral",
    10: "upper outer",
    11: "upper outer",
}

_LEGACY_LOCATION_MAP: dict[str, str | None] = {
    "medial side (inner side of the breast)": "medial",
    "upper region (upper part of the breast)": "upper",
    "lower region (lower part of the breast)": "lower",
    "upper-medial region (upper and inner side)": "upper inner",
    "subareolar region (beneath the areola/nipple)": "retroareolar",
    "middle to lateral (central to lower side)": None,
    "entire region (covers the whole breast)": None,
}

# Single-letter region codes used by TOMPEI-CMMD. The values are the surface
# forms `_LEGACY_LOCATION_MAP` above recognises, so a mapped code feeds straight
# into get_proper_location().
region_mapping = {
    "l": "lower region (lower part of the breast)",
    "m": "medial side (inner side of the breast)",
    "m-l": None,  # medial-to-lateral? middle-lower? undocumented, spans no single BI-RADS location
    "s": "subareolar region (beneath the areola/nipple)",
    "u": "upper region (upper part of the breast)",
    "u-m": "upper-medial region (upper and inner side)",
    "whole area": None,  # whole breast is not a BI-RADS location
}

# Free-text shorthands the TOMPEI-CMMD sheets use inside descriptor columns.
abbreviation_mapping = {
    "calc": "calcifications",
    "fad": "focal asymmetric density",
    "fad)": "focal asymmetric density",
    "fa": "fibroadenoma",
}


def normalize_laterality(side) -> str | None:
    if side is None or isna_v2(side):
        return None
    return LATERALITY_MAP.get(normalize_lexicon_value(side), None)


def map_clock_to_quadrant(hour: int, side) -> str | None:
    resolved = normalize_laterality(side)
    if resolved not in ("left", "right"):
        return None
    h = (12 - hour) % 12 or 12 if resolved == "left" else hour
    return CLOCK_TO_QUADRANT[h]


def get_proper_location(value, side) -> dict[str, Any]:
    out: dict[str, Any] = {
        "laterality": normalize_laterality(side),
        "clock_position": None,
        "quadrant": None,
        "special_location": None,
        "coarse_location": None,
        "depth": None,
    }

    if isna_v2(value) or value is None:
        return out

    x = normalize_lexicon_value(value)
    if not x or x == "none":
        return out

    if x in _LEGACY_LOCATION_MAP:
        legacy = _LEGACY_LOCATION_MAP[x]
        if legacy in QUADRANTS:
            out["quadrant"] = legacy
        elif legacy in SPECIAL_LOCATIONS:
            out["special_location"] = legacy
        elif legacy is not None:
            out["coarse_location"] = legacy
        return out

    parts = [p.strip() for p in re.split(r"[;,]", x) if p.strip()]
    unresolved_clock = False

    for part in parts:
        token = _normalize_for_match(part.replace("\u2019", "'"))
        token = token.replace("sub areolar", "subareolar")

        clock = CLOCK_RE.match(part.strip().replace("\u2019", "'"))
        if clock and re.search(r"o\s*'?\s*clock|:00|h|position", part):
            hour = int(clock.group("hour"))
            out["clock_position"] = hour
            quadrant = map_clock_to_quadrant(hour, side)
            if quadrant is None:
                unresolved_clock = True
            elif quadrant in QUADRANTS:
                out["quadrant"] = out["quadrant"] or quadrant
            else:
                out["coarse_location"] = out["coarse_location"] or quadrant
            continue

        if token in _DEPTH_SYNONYMS:
            out["depth"] = _DEPTH_SYNONYMS[token]
            continue

        for table, field in (
            (_SPECIAL_SYNONYMS, "special_location"),
            (_QUADRANT_SYNONYMS, "quadrant"),
            (_CARDINAL_SYNONYMS, "coarse_location"),
        ):
            if token in table:
                out[field] = out[field] or table[token]
                break

    if (
        out["clock_position"] is None
        and out["quadrant"] is None
        and out["special_location"] is None
        and out["coarse_location"] is None
        and out["depth"] is None
    ):
        if unresolved_clock:
            warn_unmapped(
                "location",
                value,
                f"a clock position cannot be projected to a quadrant without a "
                f"single-breast laterality (got {side!r})",
            )
        else:
            warn_unmapped("location", value)

    return out
