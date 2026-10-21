import glob
import logging
import os
import plistlib
import re
from collections import Counter

import numpy as np
import pandas as pd
from skimage.draw import polygon

from dtos.breast_cancer_dataset import (
    ASSESSMENT_GROUNDED,
    BreastCancerDataset,
    ExamInformation,
    stable_exam_id,
)
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext, PatientContext
from dtos.json_report_findings import (
    SCOPE_INDEX_ONLY,
    Assessment,
    BreastFindings,
    JSONReportFindings,
    Lesion,
)
from utils.error_handling import trycatch_func
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    BIRADS_MAPPING,
    CALCIFICATION,
    MAMMOGRAPHY,
    MASS,
    NOT_PRESENT,
    UNKNOWN,
    bin_age,
    collapse_known_biopsy_proven,
    csv_column_cleaning,
    dview,
    get_proper_birads,
    get_proper_breast_density,
    get_proper_exam_view,
    get_value,
    isna_v2,
    map_birads,
    normalize_laterality,
    yes_no_mapping,
)

raw_imgs_path = "../data/raw/inbreast/AllDICOMs"
raw_imgs_extension = ".dcm"
FILENAME_RE = re.compile(r"^(?P<file_id>\d+)_(?P<patient_hash>[0-9a-fA-F]{8,})_")
NON_LESION_ROI_NAMES = ("pectoral", "nipple", "skin")
segmentations_xml_folder = "../data/raw/inbreast/AllXML"
raw_segs_extension = ".xml"
clinical_data_path = "../data/raw/inbreast/INbreast.xls"
processed_imgs_path = "../data/processed/inbreast/imgs"
csv_save_path = "../data/processed/inbreast/inbreast.csv"


class Inbreast(BreastCancerDataset):
    _modality = MAMMOGRAPHY
    _machine = UNKNOWN
    _race = UNKNOWN

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(modality=MAMMOGRAPHY),
        )

        self.clinical_df = pd.read_excel(clinical_data_path)
        self.clinical_df.columns = csv_column_cleaning(list(self.clinical_df.columns))
        self.clinical_df = self.clinical_df.iloc[
            :-2
        ]  # the last two rows are totals, not needed

    def process_info(self):
        self.report.rows_seen = len(self.clinical_df)
        # Clean the clinical data
        self.clinical_df["file name"] = self.clinical_df["file name"].apply(
            lambda x: str(int(x))
        )
        self.clinical_df["lesion annotation status"] = self.clinical_df[
            "lesion annotation status"
        ].apply(lambda x: str(x).strip().lower() if pd.notna(x) else None)

        self.clinical_df["laterality"] = self.clinical_df["laterality"].apply(
            normalize_laterality
        )
        self.clinical_df["view"] = self.clinical_df["view"].apply(
            lambda x: get_value(x, dview)
        )
        self.clinical_df = self.clinical_df[
            self.clinical_df["view"] != "from below"
        ].reset_index(
            drop=True
        )  # only one exam with this view, and it's not clear what it means, so dropping it

        self.clinical_df["bi-rads"] = self.clinical_df["bi-rads"].apply(
            lambda x: map_birads(x, source="INbreast", modality=MAMMOGRAPHY)
        )
        self.clinical_df = self.clinical_df[self.clinical_df["bi-rads"].notna()]
        self.clinical_df["mass"] = self.clinical_df["mass"].apply(
            lambda x: get_value(x, yes_no_mapping) if pd.notna(x) else None
        )
        self.clinical_df["micros"] = self.clinical_df["micros"].apply(
            lambda x: get_value(x, yes_no_mapping) if pd.notna(x) else None
        )
        self.clinical_df["distortion"] = self.clinical_df["distortion"].apply(
            lambda x: (
                get_value(x, yes_no_mapping)
                if pd.notna(x) and len(str(x).strip().lower()) > 0
                else None
            )
        )
        self.clinical_df["asymmetry"] = self.clinical_df["asymmetry"].apply(
            lambda x: get_value(x, yes_no_mapping) if pd.notna(x) else None
        )
        self.clinical_df["pectoral muscle annotation"] = self.clinical_df[
            "pectoral muscle annotation"
        ].apply(
            lambda x: (
                str(x).strip().lower()
                if pd.notna(x) and len(str(x).strip().lower()) > 0
                else None
            )
        )

        # move birads 6 (known biopsy proven) to birads 5
        self.clinical_df["bi-rads"] = self.clinical_df["bi-rads"].apply(
            collapse_known_biopsy_proven
        )

        # fill mass, micros, distortion, asymmetry nan values with "no"
        self.clinical_df.fillna(
            {"mass": "no", "micros": "no", "distortion": "no", "asymmetry": "no"},
            inplace=True,
        )

        # Keep the cases whose finding we can state unambiguously: at most one of
        # mass / microcalcifications, so the lesion type is never in doubt. Rows
        # with neither are genuine normals and are exactly the negative evidence
        # the corpus needs.
        has_mass = self.clinical_df["mass"] == "yes"
        has_micros = self.clinical_df["micros"] == "yes"

        self.clinical_df = self.clinical_df[
            ~(has_mass & has_micros)
            & (self.clinical_df["distortion"] == "no")
            & (self.clinical_df["asymmetry"] == "no")
        ].reset_index(drop=True)

        # column renameing
        self.clinical_df.rename(
            columns={
                "acr": "breast density",
                "micros": "has microcalcifications",
                "view": "exam view",
            },
            inplace=True,
        )

        # Establish the grouping key used for patient-disjoint splitting
        self.resolve_patient_id()

        self.run_row_pool(self.clinical_df)

    @trycatch_func
    def process_row(self, row):
        exam_id = stable_exam_id(self.get_dataset_name(), row["file name"])
        patient_id = f"{self.get_dataset_name()}-{row['patient key']}"

        exam_imgs_path, img_shape = self.find_and_save_slice(
            row["file name"], exam_id, patient_id
        )
        segmentation_path = (
            self.create_and_save_segmentation(
                exam_id, patient_id, img_shape, row["file name"]
            )
            if not isna_v2(exam_imgs_path)
            else None
        )

        if isna_v2(exam_imgs_path):
            logging.warning(
                f"Image processing failed for patient {patient_id}, exam {row['file name']}. Skipping exam."
            )
            return None

        # construct context
        context = JSONReportContext(
            patient_context=PatientContext(age=bin_age(row.get("patient age"))),
            exam_context=ExamContext(
                modality=self._modality,
                laterality=row["laterality"],
                view=get_proper_exam_view(row["exam view"]),
            ),
        )

        # construct final exam info
        assessment_birads = get_proper_birads(row["bi-rads"], modality=MAMMOGRAPHY)

        findings = JSONReportFindings.build(
            breast=BreastFindings(
                composition=get_proper_breast_density(row["breast density"])
            ),
            lesion=(
                NOT_PRESENT
                if row["bi-rads"] == BIRADS_MAPPING[1]
                else self.get_simple_lesion(row)
            ),
            assessment=Assessment(birads=assessment_birads),
            annotation_scope=SCOPE_INDEX_ONLY,
        )

        return ExamInformation(
            id=exam_id,
            patient=patient_id,
            dataset=self.get_dataset_name(),
            modality=self._modality,
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
            breast_id=f"{patient_id}-{row['laterality']}",
            assessment_scope=ASSESSMENT_GROUNDED,
        )

    def get_simple_lesion(self, row):
        if row["mass"] == "yes":
            return Lesion(finding_type=MASS)
        elif row["has microcalcifications"] == "yes":
            return Lesion(finding_type=CALCIFICATION)
        return NOT_PRESENT

    @staticmethod
    def build_patient_hash_index() -> dict[str, str]:
        index: dict[str, str] = {}
        for path in glob.glob(
            f"{raw_imgs_path}/**/*{raw_imgs_extension}", recursive=True
        ):
            match = FILENAME_RE.match(os.path.basename(path))
            if match:
                index[match.group("file_id")] = match.group("patient_hash")

        if not index:
            return {}

        distinct = len(set(index.values()))
        if distinct >= len(index):
            logging.warning(
                f"INbreast: the DICOM filename hash is unique per image "
                f"({distinct} hashes for {len(index)} images), so it does not identify patients."
            )
            return {}

        group_sizes = Counter(Counter(index.values()).values())
        logging.info(
            f"INbreast: recovered {distinct} patient hashes for {len(index)} images "
            f"(images per patient: {dict(sorted(group_sizes.items()))})."
        )
        return index

    def resolve_patient_id(self) -> None:
        raw_ids = self.clinical_df["patient id"].astype(str).str.strip().str.lower()
        unusable = raw_ids.isin({"removed", "nan", "", "none"})

        if unusable.all():
            hashes = self.build_patient_hash_index()
            recovered = self.clinical_df["file name"].map(hashes)

            if recovered.notna().all():
                logging.info(
                    "INbreast: 'patient id' is redacted in this release; grouping by the "
                    "per-patient hash in the DICOM filenames instead."
                )
                self.clinical_df["patient key"] = recovered
                return

            logging.warning(
                "INbreast: the 'patient id' column is redacted in this release and the "
                "DICOM filenames yield no usable patient hash "
                f"({int(recovered.isna().sum())} of {len(recovered)} rows unmatched), so "
                "unresolved images are grouped together and restricted to training."
            )
            self.clinical_df["patient key"] = "unresolved"
            return

        if unusable.any():
            logging.warning(
                f"INbreast: {int(unusable.sum())} rows have no usable patient id; "
                "grouping those rows as unresolved and restricting them to training."
            )

        self.clinical_df["patient key"] = np.where(
            unusable, "unresolved", self.clinical_df["patient id"]
        )

    def find_and_save_slice(self, file_name, exam_id, patient_id):
        available_imgs = sorted(
            glob.glob(
                f"{self.image_processor.raw_imgs_path}/**/*{raw_imgs_extension}",
                recursive=True,
            )
        )
        available_imgs = [
            img
            for img in available_imgs
            if os.path.basename(img).startswith(f"{file_name}_")
        ]
        if len(available_imgs) == 0:
            logging.warning(f"Could not find image for {file_name}")
            return None, None

        if len(available_imgs) > 1:
            logging.warning(
                f"Found multiple images for {file_name}, using the first one"
            )

        img_path = available_imgs[0]
        img = self.image_processor.read_image(img_path)

        if isna_v2(img):
            logging.warning(f"Could not read image for {file_name}")
            return None, None

        img_shape = img.shape
        img = self.image_processor.apply_processing(img, is_segmentation=False)

        if isna_v2(img):
            logging.warning(f"Could not process image for {file_name}")
            return None, None

        img_save_path = self.image_processor.save_image_set([img], exam_id, patient_id)
        return img_save_path, img_shape

    def create_and_save_segmentation(self, exam_id, patient_id, img_shape, file_name):
        search_path = f"{segmentations_xml_folder}/**/*{raw_segs_extension}"
        available_imgs = sorted(glob.glob(search_path, recursive=True))
        available_imgs = [
            img
            for img in available_imgs
            if os.path.splitext(os.path.basename(img))[0] == str(file_name)
        ]

        if len(available_imgs) == 0:
            logging.info(
                f"Could not find segmentation for {file_name} in {search_path}"
            )
            return None

        if len(available_imgs) > 1:
            logging.info(
                f"Found multiple segmentations for {file_name}, using the first one"
            )

        seg_path = available_imgs[0]
        seg_mask = Inbreast.load_inbreast_mask(seg_path, img_shape)

        if isna_v2(seg_mask):
            logging.warning(f"Could not process segmentation for {file_name}")
            return None

        seg_mask = self.image_processor.apply_processing(seg_mask, is_segmentation=True)
        seg_mask_path = self.image_processor.save_segmentation_set(
            [seg_mask], exam_id, patient_id
        )
        return seg_mask_path

    @staticmethod
    @trycatch_func
    def load_inbreast_mask(mask_path, imshape):
        # taken from: https://www.kaggle.com/code/lethanhnghia/breastcancercnn
        def load_point(point_string):
            x, y = tuple([float(num) for num in point_string.strip("()").split(",")])
            return y, x

        mask = np.zeros(imshape)
        with open(mask_path, "rb") as mask_file:
            plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)["Images"][0]
            numRois = plist_dict["NumberOfROIs"]
            rois = plist_dict["ROIs"]
            assert len(rois) == numRois
            for roi in rois:
                roi_name = str(roi.get("Name", "")).strip().lower()
                if any(skip in roi_name for skip in NON_LESION_ROI_NAMES):
                    logging.debug(
                        f"Skipping '{roi_name}' ROI in {os.path.basename(mask_path)}."
                    )
                    continue

                numPoints = roi["NumberOfPoints"]
                points = roi["Point_px"]
                assert numPoints == len(points)
                points = [load_point(point) for point in points]
                if len(points) <= 2:
                    for point in points:
                        mask[int(point[0]), int(point[1])] = 1
                else:
                    x, y = zip(*points)
                    x, y = np.array(x), np.array(y)
                    poly_x, poly_y = polygon(x, y, shape=imshape)
                    mask[poly_x, poly_y] = 1
        return mask
