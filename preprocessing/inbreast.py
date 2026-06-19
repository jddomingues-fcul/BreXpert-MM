import glob
import logging
import os
import plistlib
import uuid
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from skimage.draw import polygon
from tqdm import tqdm

from dtos.breast_cancer_dataset import (
    BreastCancerDataset,
    ExamInformation,
)
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext
from dtos.json_report_findings import (
    Assessment,
    BreastFindings,
    JSONReportFindings,
    Lesion,
)
from utils.error_handling import trycatch_func
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    CALCIFICATION,
    MASS,
    NOT_PRESENT,
    UNKNOWN,
    birads_assessment,
    birads_mapping,
    breast_density,
    csv_column_cleaning,
    dview,
    get_proper_birads,
    get_proper_breast_density,
    get_proper_exam_view,
    get_value,
    isna_v2,
    laterality,
    yes_no_mapping,
)

raw_imgs_path = "../data/raw/inbreast/AllDICOMs"  # NOTE: there will not be more than 1 image per patient
raw_imgs_extension = ".dcm"
segmentations_xml_folder = "../data/raw/inbreast/AllXML"
raw_segs_extension = ".xml"
clinical_data_path = "../data/raw/inbreast/INbreast.xls"
processed_imgs_path = "../data/processed/inbreast/imgs"
csv_save_path = "../data/processed/inbreast/inbreast.csv"


class Inbreast(BreastCancerDataset):
    _modality = "mg"
    _machine = UNKNOWN
    _race = UNKNOWN

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(),
        )

        self.clinical_df = pd.read_excel(clinical_data_path)
        self.clinical_df.columns = csv_column_cleaning(list(self.clinical_df.columns))
        self.clinical_df = self.clinical_df.iloc[
            :-2
        ]  # the last two rows are totals, not needed

    def process_info(self):
        # Clean the clinical data
        self.clinical_df["file name"] = self.clinical_df["file name"].apply(
            lambda x: str(int(x))
        )
        self.clinical_df["acr"] = self.clinical_df["acr"].apply(
            lambda x: get_value(x, breast_density)
        )  # the empty strings will be converted to None
        self.clinical_df["lesion annotation status"] = self.clinical_df[
            "lesion annotation status"
        ].apply(lambda x: str(x).strip().lower() if pd.notna(x) else None)

        self.clinical_df["laterality"] = self.clinical_df["laterality"].apply(
            lambda x: get_value(x, laterality)
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
            lambda x: get_value(x, birads_assessment)
        )
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
        self.clinical_df["bi-rads"] = self.clinical_df["bi-rads"].replace(
            {birads_mapping[6]: birads_mapping[5]}
        )

        # fill mass, micros, distortion, asymmetry nan values with "no"
        self.clinical_df.fillna(
            {"mass": "no", "micros": "no", "distortion": "no", "asymmetry": "no"},
            inplace=True,
        )

        # filter for cases where we either have a mass or microcalcifications, but not both at the same time, and no to distortion and asymmetry
        self.clinical_df = self.clinical_df[
            (
                (self.clinical_df["mass"] == "yes")
                ^ (self.clinical_df["micros"] == "yes")
            )
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

        # Process the exams
        n = cpu_count() - 1
        df_split = np.array_split(self.clinical_df, n)
        with Pool(processes=n) as p:
            results = p.map(self.process_small_batch, df_split)

        for result in results:
            for exam in result:
                self.append_exam(exam)

    @trycatch_func
    def process_small_batch(self, df):
        curr_exams = []

        # Im sorry
        with tqdm(
            total=len(df),
            desc=f"Processing INBreast batch {df.index[0]} to {df.index[-1]}",
            unit="exam",
            ncols=100,
            position=0,
            leave=True,
        ) as pbar:
            for _, row in df.iterrows():
                exam = self.process_row(row)
                pbar.update(1)
                if not isna_v2(exam):
                    curr_exams.append(exam)
        return curr_exams

    @trycatch_func
    def process_row(self, row):
        exam_id = str(uuid.uuid4())
        patient_id = f"{self.get_dataset_name()}-{row['file name']}"

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
            exam_context=ExamContext(
                modality=self._modality,
                laterality=row["laterality"],
                view=get_proper_exam_view(row["exam view"]),
            ),
        )

        # construct final exam info
        findings = JSONReportFindings(
            breast=BreastFindings(
                density=get_proper_breast_density(row["breast density"])
            ),
            lesion=(
                NOT_PRESENT
                if row["bi-rads"] == birads_mapping[1]
                else self.get_simple_lesion(row)
            ),
            assessment=Assessment(birads=get_proper_birads(row["bi-rads"])),
        )

        return ExamInformation(
            id=exam_id,
            patient=patient_id,
            dataset=self.get_dataset_name(),
            modality=self._modality,
            birads=get_proper_birads(row["bi-rads"]),
            race=self._race,
            machine=self._machine,
            exam=exam_imgs_path,
            segmentation=(
                [segmentation_path] if not isna_v2(segmentation_path) else None
            ),
            context=context.get_string(),
            findings=findings.get_string(),
        )

    def get_simple_lesion(self, row):
        if row["mass"] == "yes":
            return Lesion(type=MASS)
        elif row["has microcalcifications"] == "yes":
            return Lesion(type=CALCIFICATION)
        else:
            return NOT_PRESENT

    def find_and_save_slice(self, file_name, exam_id, patient_id):
        available_imgs = sorted(
            glob.glob(
                f"{self.image_processor.raw_imgs_path}/**/*{raw_imgs_extension}",
                recursive=True,
            )
        )
        available_imgs = [
            img for img in available_imgs if str(file_name) in os.path.basename(img)
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
            img for img in available_imgs if str(file_name) in os.path.basename(img)
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

    @trycatch_func
    @staticmethod
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
