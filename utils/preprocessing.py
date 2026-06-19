import json
import logging
import os
import re
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import pydicom as dicom
import scipy
import SimpleITK as sitk
from scipy.signal import hilbert

from utils.error_handling import log_func_info, trycatch_func

UNKNOWN = "unknown"
NOT_PRESENT = "not_present"
BILATERAL = "bilateral"
MASS = "mass"
CALCIFICATION = "calcification"
CRANIAL_CAUDAL = "cranial caudal"
MEDIOLATERAL_OBLIQUE = "mediolateral oblique"


@trycatch_func
@log_func_info
def read_breast_image(img_path: str) -> np.ndarray:
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

    assert img is not None, f"Image {img_path} could not be read at `read_breast_image`"

    # convert to grayscale if the image is not already
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.normalize(img, None, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16U)
    return img


@trycatch_func
@log_func_info
def read_nii_gz_images(img_path: str) -> np.ndarray:
    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    img_np = sitk.GetArrayFromImage(img)
    img_np = cv2.normalize(img_np, None, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16U)
    return img_np


@trycatch_func
@log_func_info
# NOTE: specific to OASBUD dataset, not need to norm here as it is done in process_us_from_mat
def read_mat_images(img_path: str) -> np.ndarray:
    mat = scipy.io.loadmat(img_path)
    res = mat["data"][0]
    return res


@trycatch_func
@log_func_info
def process_us_from_mat(img: np.ndarray, db_threshold: int = -50) -> np.ndarray:
    # per https://github.com/tensorflow/datasets/pull/2428/files
    envelope_im = np.abs(hilbert(img))  # type: ignore
    compress_im = 20 * np.log10(envelope_im / np.max(envelope_im))
    compress_im[compress_im < db_threshold] = db_threshold
    result = compress_im.astype("float32")
    result = cv2.normalize(result, None, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16U)
    return result


@trycatch_func
@log_func_info
def load_images_from_npy(npy_path: str) -> np.ndarray:
    assert npy_path.endswith(".npy"), f"Invalid file extension for {npy_path}"
    assert os.path.exists(npy_path), f"File {npy_path} not found"
    return np.load(npy_path)


@trycatch_func
@log_func_info
def resize_breast_image(img: np.ndarray, resize_value: tuple) -> np.ndarray:
    return cv2.resize(img, resize_value)


@trycatch_func
@log_func_info
def pad_to_largest_dim(image: np.ndarray) -> np.ndarray:
    h, w = image.shape
    biggest_side = max(h, w)

    pad_h = biggest_side - h
    pad_w = biggest_side - w
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2

    image = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
    )
    return image


@trycatch_func
@log_func_info
def save_images_as_npy(save_path: str, images: list, dtype=None) -> str:
    save_path = f"{save_path}.npy"
    if dtype:
        np.save(save_path, np.array(images, dtype=dtype))
    else:
        np.save(save_path, np.array(images))
    return save_path


@trycatch_func
@log_func_info
def convert_dcm_image(img_path: str) -> np.ndarray:
    ds = dicom.dcmread(img_path)
    data = ds.pixel_array

    if data.ndim == 3:  # Check if the image has more than one channel
        data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

    if data.max() == data.min():
        raise ValueError(
            f"The DCM image {img_path} has no variation in pixel values (max == min)."
        )

    data = cv2.normalize(data, None, 0, 65535, cv2.NORM_MINMAX, cv2.CV_16U)
    return data


@trycatch_func
@log_func_info
def create_segmentation_mask_aux(mask: np.ndarray, rois):
    for roi in rois:
        if len(roi) != 0:
            if isinstance(roi[0], int):
                ymin, xmin, ymax, xmax = roi
                mask[ymin:ymax, xmin:xmax] = 255
            else:
                mask = create_segmentation_mask_aux(mask, roi)
    return mask


@trycatch_func
@log_func_info
def create_segmentation_mask(
    height: int, width: int, rois: tuple[tuple[int, int, int, int]]
) -> np.ndarray:
    # rois are tuples of (ymin, xmin, ymax, xmax)
    # Initialize a blank mask with zeros
    mask = np.zeros((height, width), dtype=np.uint16)
    return create_segmentation_mask_aux(mask, rois)


@trycatch_func
def upper_to_lower_wspace(s):
    # This regex matches positions where an uppercase letter is followed by a lowercase letter
    # or where a lowercase letter is followed by an uppercase letter.
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", s).lower()


def sanitize_age(age):
    if pd.isna(age):
        return None
    try:
        age_float = float(age)
        age_int = int(age_float)
        if age_int < 0 or age_int > 120:
            return None
        return str(age_int)
    except (ValueError, TypeError):
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


def sanitize_value(s: Any) -> Any:
    if s is None:
        return None

    if isinstance(s, (int, float)):
        return str(s)

    if isinstance(s, str):
        return sanitize_string(s)

    if isinstance(s, list):
        return [sanitize_value(item) for item in s]

    if isinstance(s, dict):
        return {k: sanitize_value(v) for k, v in s.items()}


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
    # Use only the first matching delimiter to avoid cross-contamination between groups
    delimiters = ["&", ",", " and ", " or "]
    for delim in delimiters:
        if delim in s:
            parts = [part.strip() for part in s.split(delim)]
            sanitized_parts = sorted([sanitize_string(part) for part in parts])
            s = f" {delim} ".join(sanitized_parts)
    return s


@trycatch_func
@log_func_info
def draw_rectangle_on_image(
    image: np.ndarray, x1: int, y1: int, x2: int, y2: int
) -> np.ndarray:
    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)
    return image


def isna_v2(value: Any) -> bool:
    na = pd.isna(value)
    if isinstance(na, bool):
        return na

    if hasattr(na, "__iter__"):
        return all(isna_v2(v) for v in na)

    return False


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


# ---------------------------------------------------------------------------
# MEDICAL MAPPINGS AND NORMALIZATION LOGIC
# ---------------------------------------------------------------------------
def get_value(value: Any, map_set: dict) -> Optional[str]:
    if value is None:
        return None
    return map_set.get(str(value).strip().lower(), None)


def get_value_default(value: Any, map_set: dict) -> Any:
    if value is None:
        return None

    default_value = value.strip().lower() if isinstance(value, str) else value
    return map_set.get(str(value).strip().lower(), default_value)


def get_recurrence_score(x: Any) -> Any:
    if x is None:
        return None

    try:
        score = int(x)
        if score < 11:
            return "low"
        if 11 <= score <= 25:
            return "intermediate"
        if score > 25:
            return "high"

    except ValueError:
        return x


def adjust_rsii(x: Any) -> Optional[str]:
    if x is None:
        return None

    if isinstance(x, int) or isinstance(x, float):
        if x == -1:
            return None

        return "positive" if x > 0.5 else "negative"

    return get_value(x, rsii)


def adjust_ki67(x: Any) -> Optional[str]:
    if x is None:
        return None

    if isinstance(x, int):
        if x == -1:
            return None

        if x <= 15:
            return "low"
        elif 16 <= x <= 30:
            return "intermediate"

        return "high"

    return get_value(x, ki67)


def get_pos_coord_slice(x: Any) -> Optional[str]:
    if x is None:
        return None

    if isinstance(x, str):
        return x[1:]
    return None


def get_oncotype_score(x: Any) -> Any:
    if x is None:
        return None

    try:
        score = int(x)
        if score < 11:
            return "low"
        if 11 <= score <= 25:
            return "intermediate"
        if score > 25:
            return "high"

    except ValueError:
        return x


birads_mapping = {
    0: "additional evaluation",
    1: "negative",
    2: "benign",
    3: "probably benign",
    4: "suspicious",
    5: "highly suggestive of malignancy",
    6: "known biopsy proven",
}

birads_mapping_with_unknown = {
    **birads_mapping,
    10: UNKNOWN,
}

birads_assessment_reverse = {v: k for k, v in birads_mapping.items()}

birads_assessment_reverse_with_unknown = {
    **birads_assessment_reverse,
    UNKNOWN: 10,
}

birads_assessment = {
    "**no assessment**": None,
    "0": "additional evaluation",
    "1": "negative",
    "normal": "negative",
    "negative": "negative",
    "2": "benign",
    "benign": "benign",
    "benign_without_callback": "benign",
    "3": "probably benign",
    "probably benign": "probably benign",
    "suspicious abnormality": "suspicious",
    "4": "suspicious",
    "4a": "suspicious",
    "4b": "suspicious",
    "4c": "suspicious",
    "malignant": "highly suggestive of malignancy",
    "highly suggestive of malignancy": "highly suggestive of malignancy",
    "5": "highly suggestive of malignancy",
    "6": "known biopsy proven",
    "known biopsy-proven malignancy": "known biopsy proven",
    "3, 4a": "suspicious",
}

modality = {
    "mr": "mr",
    "us": "us",
    "mg": "mg",  # digital mammography (dm) englobes mammography (mg), 2D and cview (cases from embed)
    "ct": "tomo",  # ea1141 cases
    "pt": "pt",
    "tomo": "tomo",
    "digital mammography": "mg",
    "contrast enhanced spectral mammography": "mg",
    "dm": "mg",
    "cesm": "mg",
    "dbt": "tomo",
}

modality_mapping = {"mr": 0, "us": 1, "mg": 2, "tomo": 3}
modality_mapping_reverse = {v: k for k, v in modality_mapping.items()}

breast_density = {
    "the breasts are almost entirely fat": "the breasts are almost entirely fat",
    "scattered fibroglandular densities": "scattered fibroglandular densities",
    "heterogeneously dense": "heterogeneously dense",
    "extremely dense": "extremely dense",
    "normal": "normal",
    "1": "the breasts are almost entirely fat",
    "a": "the breasts are almost entirely fat",
    "2": "scattered fibroglandular densities",
    "b": "scattered fibroglandular densities",
    "3": "heterogeneously dense",
    "c": "heterogeneously dense",
    "4": "extremely dense",
    "d": "extremely dense",
    "0": "normal",
    "fatty": "the breasts are almost entirely fat",
    "heterogeneous dense": "heterogeneously dense",
    "scattered": "scattered fibroglandular densities",
}

# rename laterality and diagnose_view
laterality = {
    "r": "right",
    "l": "left",
    "right": "right",
    "left": "left",
}

# sources: https://dicom.nema.org/medical/dicom/2023c/output/chtml/part16/sect_CID_4015.html
dview = {
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

ki67 = {
    "high prolif": "high proliferation rate",
    "intermed prolif": "intermediate proliferation",
    "low prolif": "low proliferation",
    "pos (high prolif rate)": "positive (high proliferation rate)",
    "pos (strong)": "positive (strong proliferation rate)",
    "pos hi prolif (49% nucs)": "positive high proliferation rate (49% nucs)",
    "intermed prolif rate": "intermediate proliferation rate",
    "high": "high",
    "low": "low",
    "intermediate": "intermediate",
    "low to intermediate": "low to intermediate",
    "intermediate to high": "intermediate to high",
    "high to intermediate": "intermediate to high",
    "60 to 70": "high",
    "30 to 40": "high",
    "15 to 20": "low to intermediate",
    "10 to 20": "low to intermediate",
    "3 to 5": "low",
    "2 to 3": "low",
    "5 to 10": "low",
    "20 to 25": "intermediate",
    "6 to 9": "low",
    "50 to 60": "high",
}

rsii = {
    "pos": "positive",
    "positive": "positive",
    "neg": "negative",
    "neg (stain moderate)": "stain moderate negative",
    "pos (strong)": "strong positive",
    "pos (strongly)": "strong positive",
    "weak": "weak positive",
    "weak positive": "weak positive",
    "pos (weak)": "weak positive",
    "neg (weak)": "weak negative",
    "pos (2+)": "positive 2+",
    "moderate by fish": "moderate by fish (fluorescence in situ hybridization)",
    "strong": "strong positive",
    "strong positive": "strong positive",
    "moderate to strong": "moderate to strong positive",
    "moderate to strong positive": "moderate to strong positive",
    "moderate": "moderate positive",
    "moderate positive": "moderate positive",
    "intermediate": "moderate positive",
    "weak to moderate": "weak to moderate positive",
    "negative": "negative",
}

referral_reasons = {
    "1": "assessment of extent of disease (known tumor/s)",
    "2": "high risk follow up - family history",
    "3": "high risk follow up - previous breast cancer",
    "4": "high risk follow up - brca",
    "5": "investigation of lesion previously seen in mammography / us / self-exam",
    "6": "post treatment - response to therapy assessment (nat)",
    "only 6 min delay": "only 6 min delay",
    "severe motion in delayed series": "severe motion in delayed series",
}

yes_no_mapping = {"0": "no", "1": "yes", "x": "yes", "y": "yes", "n": "no"}

tumor_benign_col = {
    "1": "tumor determined from biopsy/surgery pathological results",
    "0": "benign determined from biopsy, or there was a followup of at least 1 year",
}

tumor_pathology_values = {
    # malign
    "1": "idc - invasive ductal carcinoma",
    "2": "ilc- infiltrating lobular carcinoma",
    "3": "idc+dcis",
    "4": "ilc+lcis",
    "5": "carcinoma - type unspecified",
    "6": "dcis - ductal carcinoma in situ (high risk)",
    "7": "lcis (high risk)",
    "8": "adh/alh - atypical ductal/lobular hyperplasia (high risk)",
    "9": "intraductal papillary lesion (high risk)",
    "10": "metaplastic carcinoma",
    "24": "liposarcoma",
    "25": "radial scar (high risk)",
    # benign
    "11": "fibroadenoma",
    "12": "fcc - fibrocystic/fibroadenomatic changes",
    "13": "inflamation",
    "14": "healing fat necrosis",
    "15": "not seen in imri guided biopsy",
    # NOTE: Ultimately unused because we are focusing on confirmed benign lesions
    "16": "hemangioma",
    "17": "clinically benign",
    # "not sent to biopsy with confirmation 1 year follow up", => NOTE: THIS IS THE ORIGINAL VALUE
    "18": "inflammatory cyst",
    "19": "fibrotic breast tissue",
    "20": "reaction to foreign body",
    "21": "lactational changes",
    "22": "breast tissue without significant changes",
    "23": "ductal hyperplasia (usual)",
}

grade_values = {
    "1": "grade 1 or low grade (in cases such as dcis)",
    "2": "grade 2 or intermediate grade (in cases such as dcis)",
    "3": "grade 3 or high grade (in cases such as dcis)",
    "1 to 2": "1 to 2",
    "2 to 3": "2 to 3",
    "2 to3": "2 to 3",
}

pos_neg_mapping = {"0": "negative", "1": "positive"}

mammaprint_70_gene_risk = {"0": "low risk", "1": "high risk"}

race_mappings = {
    "caucasian or white": "white",
    "caucasian": "white",
    "african american": "black",
    "native american": "american indian",
    "african american  or black": "black",
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

region_mapping = {
    "l": "lower region (lower part of the breast)",
    "m": "medial side (inner side of the breast)",
    "m-l": "middle to lateral (central to lower side)",
    "s": "subareolar region (beneath the areola/nipple)",
    "u": "upper region (upper part of the breast)",
    "u-m": "upper-medial region (upper and inner side)",
    "whole area": "entire region (covers the whole breast)",
}

abbreviation_mapping = {
    "calc": "calcifications",
    "fad": "focal asymmetric density",
    "fad)": "focal asymmetric density",
    "fa": "fibroadenoma",
}

MENOPAUSE_STATUS_MAP = {
    "no prior bilateral ovariectomy  and  not on estrogen replacement  and  pre < 6 months since lmp": "pre",
    "no prior bilateral ovariectomy  and  not on estrogen replacement  and  pre <6 months since lmp": "pre",
    "no prior bilateral ovariectomy  and  not on estrogen replacement  and  peri 6-12 months since lmp": "peri",
    "pre": "pre",
    "peri": "peri",
    "post": "post",
}


# ---------------------------------------------------------------------------
# ACR BIRADS v5 LEXICON CLEANING LOGIC
# ---------------------------------------------------------------------------
def get_proper_mass_shape(value) -> Optional[str]:
    if isna_v2(value):
        return None

    x = value.strip().lower()

    valid_values = [
        "oval",
        "lobulated",
        "round",
        "irregular",
    ]

    valid_typos = {
        # ── Single legacy/typo terms ──────────────────────────────────
        "lobular": "lobulated",
        # ── Non-mass findings appearing alone ─────────────────────────
        # These are separate BI-RADS categories, not mass shapes.
        # Flagged as UNKNOWN but logged distinctly for investigation.
        "architectural_distortion": None,
        "architectural distortion": None,
        "questioned architectural distortion": None,
        "asymmetric_breast_tissue": None,
        "focal_asymmetric_density": None,
        "lymph_node": None,
        "lumph node": None,
        "intramammary lymph node": None,
        "asymmetry": None,
        "global asymmetry": None,
        "generic": None,
        # ── Compound mass shapes (most suspicious wins) ────────────────
        # irregular > lobulated > round > oval
        "lobulated-irregular": "irregular",
        "round-irregular-architectural_distortion": "irregular",  # irregular dominates
        "irregular-architectural_distortion": "irregular",  # irregular dominates; flag source record
        "irregular-focal_asymmetric_density": "irregular",  # irregular dominates; flag source record
        "irregular-asymmetric_breast_tissue": "irregular",  # irregular dominates; flag source record
        "round-lobulated": "lobulated",
        "lobulated-oval": "lobulated",
        "oval-lobulated": "lobulated",  # order variant
        "lobulated-architectural_distortion": "lobulated",  # flag source record
        "lobulated-lymph_node": "lobulated",  # flag source record
        "oval-lymph_node": "oval",  # flag source record
        "round-oval": "round",
        "irregular-oval": "irregular",
        "irregular - round": "irregular",
        "round- irregular": "irregular",
        "round - oval": "oval",
        "oval- rounded": "oval",
        "oval - round": "oval",
        "polygonal": "irregular",  # synonym
        "round/oval": "round",  # common shorthand, round dominates
        "focal asymetry": None,  # misspelling of focal asymmetric density, which is not a mass shape; flag source record
        "asymmectric tubular structure/solitary dilated duct": None,
        "asymmetric tubular structure/solitary dilated duct": None,
        "developing asymmestry": None,
    }

    if x in valid_values:
        return x

    if x in valid_typos.keys():
        return valid_typos[x]

    logging.warning(f"Value {value} not recognized for mass shape, returning None")
    return None


def get_proper_mass_density(value):
    if isna_v2(value):
        return None

    x = value.strip().lower()

    valid_values = ["high density", "equal density", "low density", "fat containing"]

    valid_typos = {
        "low density/high density": "high density",
        "high density/equal density": "high density",
        "equal": "equal density",
        "low": "low density",
        "high": "high density",
        "high with overlying microcalcification": "high density",  # flag source record
        "equal with overlying macrocalcification": "equal density",  # flag source record
        "mixed": "fat containing",  # flag source record
        "equal- overlying calcifications": "equal density",  # flag source record
        "high- overlying microcalcifications": "high density",  # flag source record
        "equal": "equal density",  # flag source record
        "mixed equal and low": "fat containing",  # flag source record
        "high - equal": "high density",  # flag source record
        "high-equal": "high density",  # flag source record
        "fat containing-radiolucent": "fat containing",  # flag source record
        "isodense": "equal density",  # synonym
        "isodense mass": "equal density",
        "equal density/isodense": "equal density",
    }

    if x in valid_values:
        return x

    if x in valid_typos.keys():
        return valid_typos[x]

    logging.warning(f"Value {value} not recognized for mass density, returning None")
    return None


def get_proper_mass_margin(value) -> Optional[str]:
    if isna_v2(value):
        return None

    x = value.strip().lower()

    valid_values = [
        "circumscribed",
        "obscured",
        "indistinct",
        "spiculated",
    ]

    valid_typos = {
        "microlobulated/indistinct/spiculated": "spiculated",
        "microlobulated/spiculated": "spiculated",
        "microlobulated/circumscribed": "indistinct",
        "speculated": "spiculated",
        "circumscribed": "circumscribed",
        "not circumscribed - indistinct": "indistinct",
        "not circumscribed -indistinct": "indistinct",
        "not circumscribed - angular&indistinct": "indistinct",
        "not circumscribed - spiculated&indistinct": "spiculated",
        "not circumscribed - microlobulated&indistinct": "indistinct",
        "not circumscribed - microlobulated": "indistinct",
        "not circumscribed - spiculated&microlobulated&indistinct": "spiculated",
        "not circumscribed - spiculated&angular&indistinct": "spiculated",
        "not circumscribed - spiculated": "spiculated",
        "not circumscribed - spiculated&angular&microlobulated&indistinct": "spiculated",
        "not circumscribed - angular": "indistinct",
        "not circumscribed - angular&microlobulated&indistinct": "indistinct",
        "not circumscribed - spiculated&angular": "spiculated",
        "not circumscribed - angular&microlobulated": "indistinct",
        "ill_defined": "indistinct",
        "microlobulated": "indistinct",
        "ill_defined-spiculated": "spiculated",
        "obscured-ill_defined-spiculated": "spiculated",
        "obscured-spiculated": "spiculated",
        "microlobulated-spiculated": "spiculated",
        "microlobulated-ill_defined-spiculated": "spiculated",
        "circumscribed-spiculated": "spiculated",
        "circumscribed-ill_defined": "indistinct",
        "obscured-ill_defined": "indistinct",
        "microlobulated-ill_defined": "indistinct",
        "circumscribed-obscured-ill_defined": "indistinct",
        "circumscribed-microlobulated-ill_defined": "indistinct",
        "microlobulated-ill_defined-spiculated": "spiculated",  # spiculated wins
        "circumscribed-microlobulated": "indistinct",
        "circumscribed-obscured": "obscured",
        "obscured-circumscribed": "obscured",  # order variant
        "partially obscured lobulated": "obscured",
        "partially obscured": "obscured",
        "lobulated partially obscured": "obscured",
        "circumscribed- obscured": "obscured",
        "speculated-circumscribed": "spiculated",
        "speculated - circumscribed": "spiculated",
        "microlobulated - circumscribed": "indistinct",
        "speculated ulcerating": "spiculated",
        "indistinct - circumscribed": "indistinct",
        "circumscribed/indistinct": "indistinct",
        "microlobulated/amorphous": "indistinct",
        "microlobulated/indistinct": "indistinct",
        "circumscribed/amorphous": "circumscribed",
        "indistinct/spiculated": "spiculated",
        "circumscribed/microlobulated": "indistinct",
    }

    if x in valid_values:
        return x

    if x in valid_typos.keys():
        return valid_typos[x]

    logging.warning(f"Value {value} not recognized for mass margin, returning None")
    return None


def contains_any(value, targets):
    for v in targets:
        if v in value:
            return v
    return None


def get_proper_halo(value) -> Optional[str]:
    if isna_v2(value):
        return None

    x = value.strip().lower()

    valid_values = ["no", "yes"]

    valid_typos = {
        "present": "yes",
        "absent": "no",
    }

    if x in valid_values:
        return x

    if x in valid_typos.keys():
        return valid_typos[x]

    logging.warning(f"Value {value} not recognized for halo, returning None")
    return None


def get_proper_posterior_features(value) -> Optional[str]:
    if isna_v2(value):
        return None

    x = value.strip().lower()

    valid_values = ["no posterior features", "enhancement", "shadowing"]

    valid_typos = {
        "present": "yes",
        "absent": "no",
        "no": "no posterior features",
        "combined": None,
    }

    if x in valid_values:
        return x

    if x in valid_typos.keys():
        return valid_typos[x]

    logging.warning(
        f"Value {value} not recognized for posterior features, returning None"
    )
    return None


def get_proper_mass_echotexture(value) -> Optional[str]:
    if isna_v2(value):
        return None

    x = value.strip().lower()

    valid_values = [
        "anechoic",
        "hyperechoic",
        "complex cystic and solid",
        "hypoechoic",
        "isoechoic",
        "heteroechoic",
    ]

    valid_typos = {
        "heterogeneous": "heteroechoic",
        "complex cystic/solid": "complex cystic and solid",
    }

    if x in valid_values:
        return x

    if x in valid_typos.keys():
        return valid_typos[x]

    logging.warning(
        f"Value {value} not recognized for mass echotexture, returning None"
    )
    return None


def get_proper_calcification_type(value) -> tuple[Optional[str], Optional[str]]:
    if isna_v2(value):
        return None, None

    x = value.strip().lower()

    typically_benign_values = [
        "skin",
        "vascular",
        "coarse",
        "large rod like",
        "round",
        "rim",
        "layering",
        "suture",
    ]
    typically_benign_typos = {
        # ── Single legacy terms ──────────────────────────────────────────
        "punctate": "round",
        "round_and_regular": "round",
        "lucent_center": "rim",
        "lucent_centered": "rim",  # spelling variant
        "eggshell": "rim",
        "dystrophic": "coarse",
        "large_rodlike": "large rod like",
        # ── Compound benign-only (most specific/prominent wins) ──────────
        "round_and_regular-lucent_center": "round",
        "round_and_regular-lucent_centered": "round",
        "round_and_regular-eggshell": "round",
        "round_and_regular-punctate": "round",
        "punctate-round_and_regular": "round",  # order variant
        "round_and_regular-lucent_center-punctate": "round",
        "lucent_center-punctate": "rim",
        "punctate-lucent_center": "rim",
        "coarse-round_and_regular": "coarse",
        "coarse-lucent_center": "coarse",
        "coarse-round_and_regular-lucent_center": "coarse",
        "coarse-round_and_regular-lucent_centered": "coarse",  # spelling variant
        "large_rodlike-round_and_regular": "large rod like",
        "round_and_regular-lucent_center-dystrophic": "coarse",  # dystrophic→coarse dominates
        "skin-punctate": "skin",
        "skin-punctate-round_and_regular": "skin",
        "skin-coarse-round_and_regular": "coarse",
        "vascular-coarse": "vascular",
        "vascular-coarse-lucent_centered": "vascular",
        "vascular-coarse-lucent_center-round_and_regular-punctate": "vascular",
        "small round calcification": "round",
        "lucent-centered": "rim",
        "lucent_centered": "rim",
        "lucent_center": "rim",
        "large rodlike": "large rod like",
        "large_rodlike": "large rod like",
        "dystrophic": "coarse",
        "milk of calcium": "layering",  # best BI-RADS fit
        "oil cyst": "rim",  # best BI-RADS fit
        "course popcorn-like": "coarse",  # typo + benign harmonization
        "coarse popcorn-like": "coarse",
        "benign": None,  # too generic to be a morphology
        "generic": None,  # too generic to be a morphology
        "punctate,round": "round",
        "milk of calcium,punctate": "layering",
        "milk of calcium,round": "layering",
        "milk of calcium,punctate,round": "layering",
        "milk of calcium,coarse": "layering",
        "milk of calcium,round,coarse": "layering",
        "round,lucent-centered": "round",
        "round,lucent-centered,coarse": "coarse",
        "lucent-centered,coarse": "coarse",
        "dystrophic,coarse": "coarse",
        "round,coarse": "coarse",
        "rim,round": "rim",
        "rim,coarse": "coarse",
        "dystrophic,rim": "coarse",
        "dystrophic,lucent-centered": "coarse",
        "dystrophic,lucent-centered,coarse": "coarse",
        "punctate,lucent-centered": "rim",
        "rim,punctate,lucent-centered": "rim",
        "punctate,vascular": "vascular",
        "lucent-centered,vascular": "vascular",
        "vascular,coarse": "vascular",
        "large rodlike,vascular": "vascular",
        "large rodlike,round": "large rod like",
        "large rodlike,round,lucent-centered": "large rod like",
        "punctate,round,lucent-centered": "round",
        "punctate,round,lucent-centered,vascular": "vascular",
        "punctate,round,skin": "skin",
        "dystrophic,oil cyst": "coarse",
        "dystrophic,rim,oil cyst": "coarse",
        "oil cyst,coarse": "coarse",
        "benign,lucent-centered": "rim",
        "benign,punctate": "round",
        "benign,round": "round",
        "benign,coarse": "coarse",
        "benign,milk of calcium": "layering",
        "benign,milk of calcium,punctate": "layering",
        "benign,milk of calcium,round": "layering",
        "benign,milk of calcium,punctate,round": "layering",
        "benign,round,lucent-centered": "round",
        "benign,skin,lucent-centered": "skin",
        "benign,oil cyst,vascular": "vascular",
        "generic,milk of calcium": "layering",
        "benign,dystrophic,oil cyst": "coarse",
        "benign,course popcorn-like,punctate,lucent-centered": "coarse",
        "punctate,coarse": "coarse",
    }

    suspicious_values = [
        "amorphous",
        "coarse heterogeneous",
        "fine pleomorphic",
        "fine linear or fine linear branching",
    ]
    suspicious_values_typos = {
        "small round/pleomorphic calcification": "fine pleomorphic",
        "amorphous and indistinct": "amorphous",
        "amorphous/pleomorphic calcification": "fine pleomorphic",
        "fine and linear calcification/pleomorphic": "fine linear or fine linear branching",
        "pleomorphic": "fine pleomorphic",
        "fine_linear_branching": "fine linear or fine linear branching",
        # fine linear branching is most suspicious
        "pleomorphic-fine_linear_branching": "fine linear or fine linear branching",
        # fine pleomorphic next
        "punctate-pleomorphic": "fine pleomorphic",
        "amorphous-pleomorphic": "fine pleomorphic",
        "coarse-pleomorphic": "fine pleomorphic",
        "round_and_regular-pleomorphic": "fine pleomorphic",
        "pleomorphic-amorphous": "fine pleomorphic",  # order variant
        "pleomorphic-pleomorphic": "fine pleomorphic",  # duplicate entry
        "punctate-amorphous-pleomorphic": "fine pleomorphic",
        "round_and_regular-punctate-amorphous-pleomorphic": "fine pleomorphic",  # hypothetical extension
        # amorphous (no higher-suspicion term present)
        "punctate-amorphous": "amorphous",
        "round_and_regular-punctate-amorphous": "amorphous",
        "round_and_regular-amorphous": "amorphous",
        "amorphous-round_and_regular": "amorphous",  # order variant
        "pleomorphic calcification": "fine pleomorphic",
        "amorphous and indistinct calcification": "amorphous",
        "fine and linear calcification": "fine linear or fine linear branching",
        "amorphous/fine and linear calcification": "fine linear or fine linear branching",
        "pleomorphic": "fine pleomorphic",
        "fine linear-branching (casting)": "fine linear or fine linear branching",
        "fine-linear (casting)": "fine linear or fine linear branching",
        "fine_linear_branching": "fine linear or fine linear branching",
        "fine-linear (casting),pleomorphic": "fine linear or fine linear branching",
        "fine linear-branching (casting),fine pleomorphic": "fine linear or fine linear branching",
        "amorphous,punctate": "amorphous",
        "amorphous,round": "amorphous",
        "amorphous,coarse": "amorphous",
        "amorphous,dystrophic": "amorphous",
        "amorphous,benign": "amorphous",
        "amorphous,milk of calcium": "amorphous",
        "amorphous,milk of calcium,punctate": "amorphous",
        "amorphous,milk of calcium,round": "amorphous",
        "amorphous,milk of calcium,coarse": "amorphous",
        "amorphous,milk of calcium,punctate,round": "amorphous",
        "amorphous,punctate,round": "amorphous",
        "amorphous,dystrophic,rim": "amorphous",
        "amorphous,skin,coarse": "amorphous",
        "amorphous,lucent-centered,vascular": "amorphous",
        "amorphous,benign,milk of calcium,punctate": "amorphous",
        "amorphous,benign,punctate": "amorphous",
        "pleomorphic,punctate": "fine pleomorphic",
        "pleomorphic,coarse": "fine pleomorphic",
        "fine pleomorphic,pleomorphic": "fine pleomorphic",
        "fine pleomorphic,punctate": "fine pleomorphic",
        "fine pleomorphic,coarse": "fine pleomorphic",
        "fine pleomorphic,punctate,coarse": "fine pleomorphic",
        "amorphous,pleomorphic": "fine pleomorphic",
        "amorphous,pleomorphic,punctate": "fine pleomorphic",
        "amorphous,fine pleomorphic": "fine pleomorphic",
        "amorphous,fine pleomorphic,punctate": "fine pleomorphic",
        "amorphous,fine-linear (casting)": "fine linear or fine linear branching",
        "coarse heterogeneous,fine linear-branching (casting)": "fine linear or fine linear branching",
        "coarse heterogeneous,fine-linear (casting)": "fine linear or fine linear branching",
        "amorphous,coarse heterogeneous,fine-linear (casting)": "fine linear or fine linear branching",
        "fine linear-branching (casting),fine pleomorphic": "fine linear or fine linear branching",
        "fine-linear (casting),pleomorphic": "fine linear or fine linear branching",
        "amorphous,coarse heterogeneous": "coarse heterogeneous",
        "coarse heterogeneous,dystrophic": "coarse heterogeneous",
        "coarse heterogeneous,milk of calcium": "coarse heterogeneous",
        "coarse heterogeneous,milk of calcium,round": "coarse heterogeneous",
        "coarse heterogeneous,milk of calcium,vascular": "coarse heterogeneous",
        "coarse heterogeneous,oil cyst,lucent-centered": "coarse heterogeneous",
        "coarse heterogeneous,lucent-centered": "coarse heterogeneous",
        "coarse heterogeneous,vascular": "coarse heterogeneous",
        "coarse heterogeneous,punctate": "coarse heterogeneous",
        "coarse heterogeneous,round": "coarse heterogeneous",
        "coarse heterogeneous,punctate,round": "coarse heterogeneous",
        "coarse heterogeneous,punctate,lucent-centered": "coarse heterogeneous",
        "coarse heterogeneous,fine pleomorphic": "fine pleomorphic",
        "amorphous,coarse heterogeneous,pleomorphic": "fine pleomorphic",
        "amorphous,coarse heterogeneous,fine pleomorphic": "fine pleomorphic",
        "amorphous,coarse heterogeneous,fine pleomorphic,milk of calcium": "fine pleomorphic",
        "amorphous,coarse heterogeneous,dystrophic": "coarse heterogeneous",
        "amorphous,coarse heterogeneous,lucent-centered": "coarse heterogeneous",
        "amorphous,coarse heterogeneous,skin": "coarse heterogeneous",
        "amorphous,coarse heterogeneous,course popcorn-like": "coarse heterogeneous",
        "course popcorn-like,punctate,vascular": "coarse heterogeneous",
    }

    if x in typically_benign_values:
        return "typically benign", x

    if x in typically_benign_typos.keys():
        return "typically benign", typically_benign_typos[x]

    if x in suspicious_values:
        return "suspicious morphology", x

    if x in suspicious_values_typos.keys():
        return "suspicious morphology", suspicious_values_typos[x]

    logging.warning(
        f"Value {value} not recognized for calcification type, returning None"
    )
    return None, None


def get_proper_exam_view(value) -> Optional[str]:
    if isna_v2(value):
        return None

    x = value.strip().lower()

    valid_values = [MEDIOLATERAL_OBLIQUE, CRANIAL_CAUDAL]

    valid_typos = {
        "cc": CRANIAL_CAUDAL,
        "mlo": MEDIOLATERAL_OBLIQUE,
    }

    if x in valid_values:
        return x

    if x in valid_typos.keys():
        return valid_typos[x]

    logging.warning(f"Value {value} not recognized for exam view, returning None")
    return None


def get_proper_calcification_distribution(value) -> Optional[str]:
    if isna_v2(value):
        return None

    x = value.strip().lower()

    valid_values = ["diffuse", "regional", "grouped", "linear", "segmental"]

    valid_typos = {
        "clustered": "grouped",
        "clustered-linear": "linear",
        "diffusely_scattered": "diffuse",
        "linear-segmental": "segmental",
        "clustered segmental": "segmental",
        "clustered-segmental": "segmental",
        "regional-regional": "regional",
        "segmental/linear": "segmental",
        "diffuse/scattered": "diffuse",
    }

    if x in valid_values:
        return x

    if x in valid_typos.keys():
        return valid_typos[x]

    logging.warning(
        f"Value {value} not recognized for calcification distribution, returning None"
    )
    return None


def get_proper_birads(value) -> str:
    if isna_v2(value):
        raise ValueError("BI-RADS value cannot be None or NaN")

    x = value.strip().lower()

    valid_values = [
        "(0) additional evaluation",
        "(1) negative",
        "(2) benign",
        "(3) probably benign",
        "(4) suspicious",
        "(5) highly suggestive of malignancy",
        "(6) known biopsy proven",
    ]

    valid_typos = {
        "additional evaluation": "(0) additional evaluation",
        "negative": "(1) negative",
        "benign": "(2) benign",
        "probably benign": "(3) probably benign",
        "suspicious": "(4) suspicious",
        "highly suggestive of malignancy": "(5) highly suggestive of malignancy",
        "known biopsy proven": "(6) known biopsy proven",
    }

    if x in valid_values:
        return x

    if x in valid_typos.keys():
        return valid_typos[x]

    raise ValueError(f"Value {value} not recognized for BI-RADS")


def get_proper_breast_density(value) -> Optional[str]:
    if isna_v2(value):
        return None

    x = value.strip().lower()

    valid_values = [
        "(a) the breasts are almost entirely fatty",
        "(b) there are scattered areas of fibroglandular density",
        "(c) the breasts are heterogeneously dense, which may obscure small masses",
        "(d) the breasts are extremely dense, which lowers the sensitivity of mammography",
    ]

    valid_typos = {
        "extremely dense": "(d) the breasts are extremely dense, which lowers the sensitivity of mammography",
        "the breasts are almost entirely fat": "(a) the breasts are almost entirely fatty",
        "scattered fibroglandular densities": "(b) there are scattered areas of fibroglandular density",
        "heterogeneously dense": "(c) the breasts are heterogeneously dense, which may obscure small masses",
        "heterogeneous dense": "(c) the breasts are heterogeneously dense, which may obscure small masses",
        "scattered": "(b) there are scattered areas of fibroglandular density",
        "fatty": "(a) the breasts are almost entirely fatty",
        "normal male": None,
        "normal": None,
    }

    if x in valid_values:
        return x

    if x in valid_typos.keys():
        return valid_typos[x]

    logging.warning(f"Value {value} not recognized for breast density, returning None")
    return None


def get_proper_tissue_composition(value) -> Optional[str]:
    if isna_v2(value):
        return None

    x = value.strip().lower()

    valid_values = [
        "heterogeneous background echotexture",
        "homogeneous background echotexture - fibroglandular",
        "Homogeneous background echotexture - fat",
    ]

    valid_typos = {
        "heterogeneous": "heterogeneous background echotexture",
        "homogeneous: fibroglandular": "homogeneous background echotexture - fibroglandular",
        "homogeneous: fat": "Homogeneous background echotexture - fat",
        "heterogeneous: predominantly fat": "heterogeneous background echotexture",
        "heterogeneous: predominantly fibroglandular": "heterogeneous background echotexture",
        "lactating&heterogeneous: predominantly fibroglandular": "heterogeneous background echotexture",
        "lactating&homogeneous: fibroglandular": "homogeneous background echotexture - fibroglandular",
        "lactating&heterogeneous: predominantly fat": "heterogeneous background echotexture",
        "lactating": None,
    }

    if x in valid_values:
        return x

    if x in valid_typos.keys():
        return valid_typos[x]

    logging.warning(
        f"Value {value} not recognized for tissue composition, returning None"
    )
    return None


def bin_age(value: Any) -> Optional[str]:
    if isna_v2(value):
        return None

    try:
        age_val = int(float(value))
        if age_val <= 39:
            return "<40"
        elif age_val < 50:
            return "40-49"
        elif age_val < 60:
            return "50-59"
        elif age_val < 70:
            return "60-69"
        elif age_val < 80:
            return "70-79"
        else:
            return "80+"
    except ValueError:
        logging.warning(
            f"Value {value} cannot be converted to an integer for age binning. Returning unknown value."
        )
        return None


def get_proper_contrast_phase(value) -> Optional[str]:
    if isna_v2(value):
        return None

    x = value.strip().lower()

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

    if x in contrast_phase_mapping.keys():
        return contrast_phase_mapping[x]
    else:
        logging.warning(
            f"Value {value} not found in contrast phase mapping. Returning unknown value."
        )
        return None


CLOCK_RE = re.compile(r"^(1[0-2]|[1-9]) o'clock$")

# ── Valid BI-RADS v2025 output terms (Section J, p.176) ──────────────────────
# Quadrant:    upper outer, upper inner, lower outer, lower inner
# Special:     retroareolar, central, axillary tail
# Cardinal:    upper, lower, medial, lateral  (less preferred but valid)

DIRECT_LOCATION_TARGETS = {
    # Special locations (Section J, p.176)
    "retroareolar": "retroareolar",
    "central": "central",
    "axillary tail": "axillary tail",
    # Quadrant locations (Section J, p.176)
    "upper outer": "upper outer",
    "upper inner": "upper inner",
    "lower outer": "lower outer",
    "lower inner": "lower inner",
    # Cardinal — less specific but acceptable per BI-RADS
    "upper": "upper",
    "superior": "upper",  # legacy synonym
    "lower": "lower",
    "inferior": "lower",  # legacy synonym
    "medial": "medial",
    "inner": "medial",  # lay synonym
    "lateral": "lateral",
    "outer": "lateral",  # lay synonym
}

# ── Clock-face to quadrant (laterality-aware) ────────────────────────────────
# Based on anatomical convention:
# Right breast: clock runs clockwise from patient perspective
# Left breast:  clock is mirrored
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

LATERALITY_MAP = {
    "left": "left",
    "right": "right",
    "bilateral": "bilateral",
    "l": "left",
    "r": "right",
    "b": "bilateral",
    "both": "bilateral",
    "left breast": "left",
    "right breast": "right",
}

# ── Legacy non-standard output strings from previous function version ────────
LEGACY_LOCATION_MAP = {
    "medial side (inner side of the breast)": "medial",
    "upper region (upper part of the breast)": "upper",
    "lower region (lower part of the breast)": "lower",
    "middle to lateral (central to lower side)": "lateral",
    "upper-medial region (upper and inner side)": "upper inner",
    "subareolar region (beneath the areola/nipple)": "retroareolar",
}


def normalize_laterality(laterality) -> str | None:
    if laterality is None or isna_v2(laterality):
        return None
    x = str(laterality).strip().lower()
    return LATERALITY_MAP.get(x, None)


def normalize_location_token(token: str) -> str:
    x = str(token).strip().lower()
    x = re.sub(r"\s+", " ", x)

    # Normalize all sub-areolar variants TO the BI-RADS standard term
    x = x.replace("subareolar", "retroareolar")
    x = x.replace("sub-areolar", "retroareolar")
    x = x.replace("sub areolar", "retroareolar")

    return x


def token_priority(token: str) -> int:
    """
    Lower number = higher specificity = preferred.
    Priority: clock > quadrant > special > cardinal
    """
    if CLOCK_RE.match(token):
        return 0
    if token in {"upper outer", "upper inner", "lower outer", "lower inner"}:
        return 1
    if token in {"retroareolar", "central", "axillary tail"}:
        return 2
    if token in {
        "upper",
        "lower",
        "medial",
        "lateral",
        "superior",
        "inferior",
        "inner",
        "outer",
    }:
        return 3
    return 99


def map_clock_to_quadrant(clock_value: str) -> Optional[str]:
    """
    Maps a clock-face position to a BI-RADS quadrant term.
    """
    h = int(clock_value.split()[0])
    return CLOCK_TO_QUADRANT[h]


DEPTH_TOKENS = {
    "anterior",
    "middle",
    "posterior",
    "anterior third",
    "middle third",
    "posterior third",
}


def get_proper_location(value) -> Optional[str]:
    if isna_v2(value) or value is None:
        return None

    x = str(value).strip().lower()
    if not x or x == "none":
        return None

    # ── Intercept legacy non-standard strings first ──────────────────────────
    if x in LEGACY_LOCATION_MAP:
        return LEGACY_LOCATION_MAP[x]

    # Split on comma, normalize each token
    parts = [normalize_location_token(p) for p in x.split(",") if p.strip()]
    if not parts:
        return None

    # remove depth-only tokens from location cleaning
    parts = [p for p in parts if p not in DEPTH_TOKENS]
    if not parts:
        return None

    # Remove duplicates, preserve order
    seen = set()
    deduped = []
    for p in parts:
        if p not in seen:
            deduped.append(p)
            seen.add(p)

    # Sort by specificity — most specific token wins
    deduped.sort(key=token_priority)

    for token in deduped:
        # Clock-face position (most preferred per BI-RADS Section J)
        if CLOCK_RE.match(token):
            mapped = map_clock_to_quadrant(token)
            if mapped is not None:
                return mapped

        # Direct location terms
        mapped = DIRECT_LOCATION_TARGETS.get(token, None)
        if mapped is not None:
            return mapped

    logging.warning(f"Value '{value}' not recognized for location, returning None")
    return None
