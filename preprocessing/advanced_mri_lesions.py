import glob
import logging
import math
import os
import random
import uuid
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from typing import Optional

import numpy as np
import pandas as pd
import pydicom as dicom
from tqdm import tqdm

from dtos.breast_cancer_dataset import BreastCancerDataset, ExamInformation
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext, PatientContext
from dtos.json_report_findings import Assessment, JSONReportFindings, Lesion
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    BILATERAL,
    MASS,
    NOT_PRESENT,
    UNKNOWN,
    bin_age,
    birads_assessment,
    birads_mapping,
    column_cleaning_csv_reading,
    csv_column_cleaning,
    get_or_default,
    get_pos_coord_slice,
    get_proper_birads,
    get_proper_contrast_phase,
    get_value,
    isna_v2,
    laterality,
    referral_reasons,
    sanitize_age,
    yes_no_mapping,
)

raw_imgs_path = "../data/raw/advanced-mri-breast-lesions"
processed_imgs_path = "../data/processed/advanced-mri-breast-lesions/imgs"
mri_lesions_path = "../data/raw/advanced-mri-breast-lesions/Advanced-MRI-Breast-Lesions-DA-Clinical-Jan112024.xlsx"
metadata_path = "../data/raw/advanced-mri-breast-lesions/metadata.csv"
csv_save_path = (
    "../data/processed/advanced-mri-breast-lesions/advanced-mri-breast-lesions.csv"
)


class AdvancedMRILesions(BreastCancerDataset):
    _modality = "mr"
    _machines = "1.5 T"
    _race = UNKNOWN

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(),
        )
        self.mri_lesions_df = pd.read_excel(mri_lesions_path, sheet_name=0, header=1)
        self.metadata_df = column_cleaning_csv_reading(metadata_path).reset_index()
        self.rng = random.Random(42)

    def process_info(self):
        # 1. Drop irrelevant columns
        self.mri_lesions_df = self.mri_lesions_df.drop(
            columns=[
                "tumor/benign1",
                "GRADE1",
                "ER [SII] 1",
                "PR [SII] 1",
                "HER2 [SII] 1",
                "isTN1",
                "ER [%] 1",
                "PR [%] 1",
                "HER2 [%] 1",
                "Unnamed: 17",
                "KI67[%] 1",
                "tumor/benign2",
                "GRADE2",
                "ER [SII] 2",
                "PR [SII] 2",
                "HER2 [SII] 2",
                "isTN2",
                "ER [%] 2",
                "PR  [%] 2",
                "HER  [%] 2",
                "Unnamed: 30",
                "KI67[%] 2",
                "tumor/benign3",
                "GRADE3",
                "ER [SII] 3",
                "PR [SII] 3",
                "HER [SII] 3",
                "isTN3",
                "ER [%] 3",
                "PR  [%] 3",
                "HER  [%] 3",
                "Unnamed: 43",
                "KI67[%] 3",
                "tumor/benign4",
                "tumor/benign5",
                "tumor/benign6",
            ]
        )

        # 2. clean mri lesions with -1 and -1.0 values
        self.mri_lesions_df = self.mri_lesions_df.replace(-1, None)
        self.mri_lesions_df = self.mri_lesions_df.replace(-1.0, None)

        # 3. adjust column names
        self.mri_lesions_df.columns = [
            col.replace("id#", "")
            for col in csv_column_cleaning(list(self.mri_lesions_df.columns))
        ]
        self.mri_lesions_df = self.mri_lesions_df.rename(
            columns={"patient id": "subject id"}
        )

        # 4. classify each pathology into mass vs None lesion type finding
        for i in range(1, 7):
            self.mri_lesions_df[f"type lesion{i}"] = self.mri_lesions_df[
                f"pathology{i}"
            ].apply(
                lambda x: (
                    PATHOLOGY_TO_MRI_TYPE.get(int(x), None) if pd.notna(x) else None
                )
            )

        # 5. clean metadata
        adjusted_columns = list(self.metadata_df.columns[1:])
        adjusted_columns.insert(adjusted_columns.index("file size"), UNKNOWN)
        self.metadata_df.columns = adjusted_columns

        # 6. only keep needed columns
        meta_cols_to_keep = [
            "subject id",
            "study date",
            "study description",
            "series description",
            "file location",
            "modality",
            "manufacturer",
        ]
        self.metadata_df = self.metadata_df[meta_cols_to_keep]

        # 7. Create a copy for MRIs and SEG
        seg_metadata_df = deepcopy(
            self.metadata_df[self.metadata_df["modality"] == "SEG"]
        ).reset_index(drop=True)
        mr_metadata_df = deepcopy(
            self.metadata_df[self.metadata_df["modality"] == "MR"]
        ).reset_index(drop=True)

        # 8. Create the segmentation paths in the segmentation metadata
        seg_metadata_df["segmentation path"] = seg_metadata_df["file location"].apply(
            lambda x: glob.glob(
                os.path.join(raw_imgs_path, x[2:], "*"), recursive=True
            )[0]
        )
        seg_metadata_df = seg_metadata_df[["subject id", "segmentation path"]]

        # 9. join both datasets
        self.mri_lesions_df = pd.merge(
            self.mri_lesions_df, mr_metadata_df, on=["subject id"], how="inner"
        )
        self.mri_lesions_df = pd.merge(
            self.mri_lesions_df, seg_metadata_df, on=["subject id"], how="left"
        )

        # 10. Remove the segmentation paths for the images that do not match metadata indication
        self.mri_lesions_df["segmentation path"] = self.mri_lesions_df.apply(lambda x: self.remove_non_mri_segmentation_paths(x), axis=1)  # type: ignore

        # 11. fill in contents
        self.fill_in_contents()

        # 12. We only save those that have a segmentation i.e. those where we actually have a lesion segmented and now where to observe it
        self.mri_lesions_df = self.mri_lesions_df[
            self.mri_lesions_df["segmentation path"].notnull()
        ].reset_index(drop=True)

        # 13. Drop any rows with birads null
        self.mri_lesions_df = self.mri_lesions_df[
            self.mri_lesions_df["birads"].notnull()
        ].reset_index(drop=True)

        # 14. drop any additional information birads
        self.mri_lesions_df = self.mri_lesions_df[
            self.mri_lesions_df["birads"] != birads_mapping[0]
        ].reset_index(drop=True)

        # 15. swap known biopsy proven to highly suggestive of malignancy
        self.mri_lesions_df["birads"] = self.mri_lesions_df["birads"].replace(
            {birads_mapping[6]: birads_mapping[5]}
        )

        # 16. converting age to int
        self.mri_lesions_df["age at mri"] = self.mri_lesions_df["age at mri"].apply(
            lambda x: sanitize_age(x)
        )

        # Process the exams
        n = cpu_count() - 1
        df_split = [
            self.mri_lesions_df.iloc[idx]
            for idx in np.array_split(np.arange(len(self.mri_lesions_df)), n)
            if len(idx) > 0
        ]

        with Pool(processes=n) as p:
            results = p.map(self.process_small_batch, df_split)

        for result in results:
            for exam in result:
                self.append_exam(exam)

    def process_small_batch(self, df):
        curr_exams = []

        with tqdm(
            total=len(df),
            desc=f"Processing advanced mri lesions batch {df.index[0]} to {df.index[-1]}",
            unit="exam",
            ncols=100,
            position=0,
            leave=True,
        ) as pbar:
            for _, row in df.iterrows():
                curr_exams.extend(self.process_row(row))
                pbar.update(1)
        return curr_exams

    def process_row(self, row):
        folder_path = row["file location"][2:]  # to remove the "./" from the beginning
        pat = str(row["subject id"])
        seg_path = row["segmentation path"]
        resulting_exams = []

        # For the moment we just consider cases with a single lesion
        for i in range(1, 7):
            has_previous = False
            has_post = False
            is_post_seg = False

            if isna_v2(row[f"slice coord lesion{i}"]):
                continue

            slice = row[f"slice coord lesion{i}"]

            searchable_folder = os.path.join(
                self.image_processor.raw_imgs_path, folder_path
            )
            available_imgs = sorted(glob.glob(f"{searchable_folder}/*", recursive=True))

            if isna_v2(seg_path) or not os.path.exists(seg_path):
                logging.warning(
                    f"Segmentation path does not exist for patient {pat} on the images at path {folder_path}. Saving images only."
                )
                seg_data = None
            else:
                seg_data = dicom.dcmread(seg_path)

            for img in available_imgs:
                if os.path.isdir(img) or not os.path.exists(img):
                    continue

                dicom_data = dicom.dcmread(img)

                if isna_v2(slice):
                    continue

                # only mass findings are allowed to be saved => currently mass is the only one, so, not none.
                if (
                    self.matches_slice(dicom_data, slice)
                    and row[f"type lesion{i}"] is not None
                ):  # rounding to 2 decimals to match the values in the provided excel
                    exam_id = str(uuid.uuid4())
                    is_post_seg = True  # from here onwards we know we are in the next set of slices after the lesion
                    curr_img = self.image_processor.process_image(img)
                    if curr_img is not None:
                        sois_path = self.image_processor.save_image_set(
                            [curr_img], exam_id, pat
                        )
                    else:
                        logging.warning(
                            f"Image processing failed for patient {pat} on the image at path {img}. Skipping exam."
                        )
                        sois_path = None

                    if seg_data is not None:
                        corresponding_seg = self.get_corresponding_seg(seg_data, slice)
                        if corresponding_seg is not None:
                            segs_path = self.image_processor.save_segmentation_set(
                                [corresponding_seg], exam_id, pat
                            )
                        else:
                            logging.warning(
                                f"No corresponding segmentation slice found for patient {pat} on the images at path {folder_path}. Skipping segmentation."
                            )
                            segs_path = None
                    else:
                        logging.warning(
                            f"No segmentation data available for patient {pat} on the images at path {folder_path}. Skipping segmentation."
                        )
                        segs_path = None

                    # construct context
                    cphase = self.get_contrast_from_img_path(img)
                    context = JSONReportContext(
                        patient_context=PatientContext(
                            age=bin_age(row["age at mri"]),
                            has_implants=get_or_default(row["breast implants"], None),
                        ),
                        exam_context=ExamContext(
                            modality=self._modality,
                            laterality=BILATERAL,
                            contrast_phase=get_proper_contrast_phase(cphase),
                        ),
                    )

                    # construct final exam info
                    findings = JSONReportFindings(
                        lesion=(
                            NOT_PRESENT
                            if row["birads"] == birads_mapping[1]
                            else self._construct_lesions_array(row, i)
                        ),
                        assessment=Assessment(birads=get_proper_birads(row["birads"])),
                    )

                    curr_exam = ExamInformation(
                        id=exam_id,
                        patient=pat,
                        dataset=self.get_dataset_name(),
                        modality=self._modality,
                        birads=get_proper_birads(row["birads"]),
                        race=self._race,
                        machine=row["manufacturer"] + " " + self._machines,
                        exam=sois_path,
                        segmentation=[segs_path] if not isna_v2(segs_path) else None,
                        context=context.get_string(),
                        findings=findings.get_string(),
                    )

                    resulting_exams.append(curr_exam)

                else:
                    if (
                        not has_previous and not is_post_seg and self.rng.random() < 0.3
                    ):  # to add some variability to the number of negative samples per exam
                        has_previous = True
                        exam_id = str(uuid.uuid4())

                        curr_img = self.image_processor.process_image(img)
                        if curr_img is not None:
                            sois_path = self.image_processor.save_image_set(
                                [curr_img], exam_id, pat
                            )
                        else:
                            logging.warning(
                                f"Image processing failed for patient {pat} on the image at path {img}. Skipping exam."
                            )
                            sois_path = None

                        # construct context
                        cphase = self.get_contrast_from_img_path(img)
                        context = JSONReportContext(
                            patient_context=PatientContext(
                                age=bin_age(row["age at mri"]),
                                has_implants=get_or_default(
                                    row["breast implants"], None
                                ),
                            ),
                            exam_context=ExamContext(
                                modality=self._modality,
                                laterality=BILATERAL,
                                contrast_phase=get_proper_contrast_phase(cphase),
                            ),
                        )

                        # construct final exam info
                        findings = JSONReportFindings(
                            lesion=NOT_PRESENT,
                            assessment=Assessment(
                                birads=get_proper_birads(birads_mapping[1])
                            ),
                        )

                        curr_exam = ExamInformation(
                            id=exam_id,
                            patient=pat,
                            dataset=self.get_dataset_name(),
                            modality=self._modality,
                            birads=get_proper_birads(birads_mapping[1]),
                            race=self._race,
                            machine=row["manufacturer"] + " " + self._machines,
                            exam=sois_path,
                            segmentation=None,
                            context=context.get_string(),
                            findings=findings.get_string(),
                        )

                        resulting_exams.append(curr_exam)

                    elif (
                        not has_post and is_post_seg and self.rng.random() < 0.3
                    ):  # to add some variability to the number of negative samples per exam
                        has_post = True
                        exam_id = str(uuid.uuid4())

                        curr_img = self.image_processor.process_image(img)
                        if curr_img is not None:
                            sois_path = self.image_processor.save_image_set(
                                [curr_img], exam_id, pat
                            )
                        else:
                            logging.warning(
                                f"Image processing failed for patient {pat} on the image at path {img}. Skipping exam."
                            )
                            sois_path = None

                        # construct context
                        cphase = self.get_contrast_from_img_path(img)
                        context = JSONReportContext(
                            patient_context=PatientContext(
                                age=bin_age(row["age at mri"]),
                                has_implants=get_or_default(
                                    row["breast implants"], None
                                ),
                            ),
                            exam_context=ExamContext(
                                modality=self._modality,
                                laterality=BILATERAL,
                                contrast_phase=get_proper_contrast_phase(cphase),
                            ),
                        )

                        # construct final exam info
                        findings = JSONReportFindings(
                            lesion=NOT_PRESENT,
                            assessment=Assessment(
                                birads=get_proper_birads(birads_mapping[1])
                            ),
                        )

                        curr_exam = ExamInformation(
                            id=exam_id,
                            patient=pat,
                            dataset=self.get_dataset_name(),
                            modality=self._modality,
                            birads=get_proper_birads(birads_mapping[1]),
                            race=self._race,
                            machine=row["manufacturer"] + " " + self._machines,
                            exam=sois_path,
                            segmentation=None,
                            context=context.get_string(),
                            findings=findings.get_string(),
                        )

                        resulting_exams.append(curr_exam)

        return resulting_exams

    def get_contrast_from_img_path(self, img) -> str:
        img_lower = img.lower()
        base = os.path.basename(img_lower)
        contrast_phase = base.split("-")[0]
        return f"post-contrast phase {contrast_phase}"

    def _construct_lesions_array(self, row, i) -> Optional[Lesion]:
        if isna_v2(row[f"laterality lesion{i}"]):
            return None
        return Lesion(laterality=row[f"laterality lesion{i}"], type=MASS)

    @staticmethod
    def remove_non_mri_segmentation_paths(row):
        # NOTE: Only these samples actually match on the segmentation when manually validating metadata so these are the ones we use
        desc = row["series description"].lower()
        if desc != "Registered AX Sen Vibrant MultiPhase".lower():
            return None
        return row["segmentation path"]

    def get_corresponding_seg(self, seg_data, slice):
        seg_index = -1
        for i, gs in enumerate(seg_data["PerFrameFunctionalGroupsSequence"].value):
            slic_coord = str(
                round(
                    gs["PlanePositionSequence"]
                    .value[0]["ImagePositionPatient"]
                    .value[-1],
                    2,
                )
            )
            if slice in slic_coord:
                seg_index = i
                break

        if seg_index == -1:
            return None

        seg_index = seg_index + 1  # 1-based index
        seg_slice = seg_data.pixel_array[seg_index]

        data_min, data_max = seg_slice.min(), seg_slice.max()

        if data_max == data_min:
            return None

        # Normalize the image for cv2
        data = self.image_processor.apply_processing(seg_slice, is_segmentation=True)
        return data

    def fill_in_contents(self):
        # fill contents of referral
        self.mri_lesions_df["reason for referral "] = self.mri_lesions_df[
            "reason for referral "
        ].apply(
            lambda x: get_value(int(x), referral_reasons) if not isna_v2(x) else None
        )
        self.mri_lesions_df["additional reason for referral "] = self.mri_lesions_df[
            "additional reason for referral "
        ].apply(lambda x: get_value(x, referral_reasons))

        # breast implants info
        self.mri_lesions_df["breast implants"] = self.mri_lesions_df[
            "breast implants"
        ].apply(lambda x: get_value(x, yes_no_mapping))

        # birads info
        self.mri_lesions_df["birads"] = self.mri_lesions_df["birads"].apply(
            lambda x: get_value(x, birads_assessment)
        )

        # positions for tumors
        for i in range(1, 7):
            self.mri_lesions_df[f"laterality lesion{i}"] = self.mri_lesions_df[
                f"pos{i}"
            ].apply(
                lambda x: get_value(str(x)[0], laterality) if not isna_v2(x) else None
            )
            self.mri_lesions_df[f"slice coord lesion{i}"] = self.mri_lesions_df[
                f"pos{i}"
            ].apply(lambda x: get_pos_coord_slice(str(x)) if not isna_v2(x) else None)
        self.mri_lesions_df = self.mri_lesions_df.drop(
            [f"pos{i}" for i in range(1, 7)], axis=1
        )

    def matches_slice(self, ds, pos_str: str, tol=1e-2):
        target = float(pos_str)
        if target is None:
            return False
        if "SliceLocation" not in ds:
            return False
        slice_loc = float(ds.SliceLocation)
        return math.isclose(slice_loc, target, abs_tol=tol)


PATHOLOGY_TO_MRI_TYPE = {
    1: "mass",
    2: None,  # "mass"   # caution: can be NME
    3: "mass",
    4: None,  # "mass",   # caution: mixed lobular biology
    5: "mass",
    6: None,  # "non-mass enhancement",
    7: None,  # "non-mass enhancement",
    8: None,  # "non-mass enhancement",
    9: "mass",
    10: "mass",
    11: "mass",
    12: None,
    13: None,
    14: None,
    15: None,
    16: "mass",
    17: None,
    18: "mass",
    19: None,
    20: None,
    21: None,
    22: None,
    23: None,
    24: "mass",
    25: None,  # "non-mass enhancement",
}
