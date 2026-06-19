import logging
import os
import uuid
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from dtos.breast_cancer_dataset import BreastCancerDataset, ExamInformation
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext, PatientContext
from dtos.json_report_findings import Assessment, BreastFindings, JSONReportFindings
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    BILATERAL,
    NOT_PRESENT,
    UNKNOWN,
    bin_age,
    birads_mapping,
    breast_density,
    csv_column_cleaning,
    get_proper_birads,
    get_proper_breast_density,
    get_proper_contrast_phase,
    get_value,
    isna_v2,
    race_mappings,
    read_nii_gz_images,
    sanitize_age,
    yes_no_mapping,
)

raw_imgs_path = "../data/raw/mama-mia/images"
raw_segs_path = "../data/raw/mama-mia/segmentations/expert"
raw_imgs_extension = ".nii.gz"
processed_imgs_path = "../data/processed/mama-mia/imgs"
clinical_data_path = "../data/raw/mama-mia/clinical_and_imaging_info.xlsx"
csv_save_path = "../data/processed/mama-mia/mama-mia.csv"


class MamaMia(BreastCancerDataset):
    _modality = "mr"
    _birads = "highly suggestive of malignancy"  # all patients have cancer (birads 6), but we set this to the highest non-biopsy proven value (birads 5)

    _SLICES_AFTER_AND_BEFORE_LESION_FOR_NEGATIVES = 1

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(
                read_func=partial(read_nii_gz_images)
            ),
        )

        self.clinical_df = pd.read_excel(clinical_data_path)
        self.clinical_df.columns = csv_column_cleaning(list(self.clinical_df.columns))

    def process_info(self):
        # Drop un-needed columns
        self.clinical_df = self.clinical_df.drop(
            columns=[
                "nac agent",
                "endocrine therapy",
                "anti her2 neu therapy",
                "pcr",
                "mastectomy post nac",
                "days to follow up",
                "days to recurrence",
                "days to metastasis",
                "days to death",
                "hr",
                "er",
                "pr",
                "her2",
                "mammaprint",
                "oncotype score",
                "nottingham grade",
                "tumor subtype",
                "patient size",
                "weight",
                "high bit",
                "window center",
                "window width",
                "field strength",
                "fat suppressed",
                "image rows",
                "image columns",
                "num slices",
                "pixel spacing",
                "slice thickness",
                "site",
                "echo time",
                "repetition time",
                "acquisition date",
                "tcia series uid",
            ]
        )

        # Adjust values
        self.clinical_df["bilateral breast cancer"] = self.clinical_df[
            "bilateral breast cancer"
        ].apply(lambda x: get_value(x, yes_no_mapping))
        self.clinical_df["multifocal cancer"] = self.clinical_df[
            "multifocal cancer"
        ].apply(lambda x: get_value(int(x), yes_no_mapping) if not isna_v2(x) else None)
        self.clinical_df["ethnicity"] = self.clinical_df["ethnicity"].apply(
            lambda x: get_value(x, race_mappings) if not isna_v2(x) else UNKNOWN
        )
        self.clinical_df["has implant"] = self.clinical_df["has implant"].apply(
            lambda x: get_value(x, yes_no_mapping)
        )
        self.clinical_df["breast density"] = self.clinical_df["breast density"].apply(
            lambda x: get_value(x, breast_density) if not isna_v2(x) else None
        )
        self.clinical_df["bilateral mri"] = self.clinical_df["bilateral mri"].apply(
            lambda x: get_value(x, yes_no_mapping)
        )
        self.clinical_df["bmi group"] = self.clinical_df["bmi group"].apply(
            lambda x: " ".join(x.split("_")) if not isna_v2(x) else None
        )

        # Split DataFrame by patient id
        grouped_patients = [
            group for _, group in self.clinical_df.groupby("patient id")
        ]

        # Process the exams
        n = cpu_count() - 1
        with Pool(processes=n) as p:
            results = list(
                tqdm(
                    p.imap(self.process_patient, grouped_patients),
                    total=len(grouped_patients),
                    desc="Processing mama-mia patients",
                    unit="patient",
                    ncols=100,
                    position=0,
                    leave=True,
                )
            )
        for result in results:
            for exam in result:
                self.append_exam(exam)

    def process_patient(self, grpo):
        curr_exams = []

        if len(grpo) > 1:
            logging.warning(
                f"Patient {grpo['patient id'].iloc[0]} has more than one row in the clinical data. Only the first row will be used."
            )

        grpo = grpo.iloc[
            0
        ]  # NOTE: The dataframe per patient will always only 1 row, so we access it this way

        available_imgs = sorted(
            os.listdir(os.path.join(raw_imgs_path, grpo["patient id"]))
        )
        available_imgs = [
            path for path in available_imgs if path.endswith(raw_imgs_extension)
        ]

        exams_contrast_phase = (
            self.get_exams_types(eval(grpo["acquisition times"]))
            if not isna_v2(grpo["acquisition times"])
            else [None] * len(available_imgs)
        )

        seg_path = os.path.join(
            raw_segs_path, f"{grpo['patient id']}{raw_imgs_extension}"
        )
        seg_slices = self.image_processor.read_image(seg_path)
        if not isna_v2(seg_slices):
            seg_slices = np.array(
                [
                    self.image_processor.apply_processing(ss, is_segmentation=True)
                    for ss in seg_slices
                ]
            )
        else:
            logging.warning(f"Segmentation {seg_path} was not able to be read")

        for i, exam_path in enumerate(available_imgs):
            exam_path = os.path.join(raw_imgs_path, grpo["patient id"], exam_path)

            if isna_v2(seg_slices):
                logging.warning(
                    f"Skipping exam {exam_path} for patient {grpo['patient id']} due to missing segmentation."
                )
                continue

            exam_slices = self.image_processor.read_image(exam_path)
            if not isna_v2(exam_slices):
                processed_slices = np.array(
                    [
                        self.image_processor.apply_processing(slice)
                        for slice in exam_slices
                    ]
                )
            else:
                logging.warning(f"Exam images of {exam_path} were not able to be read")
                continue

            # compute the indexes of the slices that have segmentation with non-zero values => lesion present
            slices_with_seg = [
                idx for idx in range(seg_slices.shape[0]) if np.sum(seg_slices[idx]) > 0
            ]

            # just keep the central slice of the slices_with_seg => avoid near-duplicate slices for the same lesion
            slices_with_seg = (
                [slices_with_seg[len(slices_with_seg) // 2]]
                if len(slices_with_seg) > 0
                else []
            )

            # construct a new slices without segmentation for negative samples
            # we take a number of slices before the lowest slice with segmentation and after the highest slice with segmentation
            slices_without_seg = []

            if len(slices_with_seg) > 0:
                lowest_slice_with_seg = (
                    min(slices_with_seg)
                    - self._SLICES_AFTER_AND_BEFORE_LESION_FOR_NEGATIVES
                )  # buffer of slices before the lesion
                highest_slice_with_seg = (
                    max(slices_with_seg)
                    + self._SLICES_AFTER_AND_BEFORE_LESION_FOR_NEGATIVES
                )  # buffer of slices after the lesion

                # slices before the lesion
                start_idx = max(
                    0,
                    lowest_slice_with_seg
                    - self._SLICES_AFTER_AND_BEFORE_LESION_FOR_NEGATIVES,
                )
                end_idx = lowest_slice_with_seg
                slices_without_seg.extend(list(range(start_idx, end_idx)))

                # slices after the lesion
                start_idx = highest_slice_with_seg + 1
                end_idx = min(
                    seg_slices.shape[0],
                    highest_slice_with_seg
                    + 1
                    + self._SLICES_AFTER_AND_BEFORE_LESION_FOR_NEGATIVES,
                )
                slices_without_seg.extend(list(range(start_idx, end_idx)))

            for k in slices_with_seg:  # we have a non-empty segmentation
                exam_id = str(uuid.uuid4())
                contrast_phase = exams_contrast_phase[i]
                imgs_path = self.image_processor.save_image_set(
                    imgs=[processed_slices[k]],
                    exam_id=exam_id + f"_{contrast_phase}_slice_{k}",
                    patient_id=grpo["patient id"],
                )
                segs_path = self.image_processor.save_segmentation_set(
                    segs=[seg_slices[k]],
                    exam_id=exam_id + f"_{contrast_phase}_slice_{k}",
                    patient_id=grpo["patient id"],
                )

                if isna_v2(imgs_path):
                    logging.warning(
                        f"Skipping exam {exam_path} for patient {grpo['patient id']} due to processing issues."
                    )
                    continue

                # construct context
                context = JSONReportContext(
                    patient_context=PatientContext(
                        age=bin_age(sanitize_age(grpo["age"])),
                        has_implants=grpo["has implant"],
                    ),
                    exam_context=ExamContext(
                        modality=self._modality,
                        laterality=(
                            BILATERAL if grpo["bilateral mri"] == "yes" else None
                        ),
                        contrast_phase=get_proper_contrast_phase(contrast_phase),
                    ),
                )

                # construct final exam info
                findings = JSONReportFindings(
                    breast=BreastFindings(
                        density=get_proper_breast_density(grpo["breast density"])
                    ),
                    assessment=Assessment(birads=get_proper_birads(self._birads)),
                )

                exam = ExamInformation(
                    id=exam_id,
                    patient=grpo["patient id"],
                    dataset=grpo["dataset"],
                    modality=self._modality,
                    birads=get_proper_birads(self._birads),
                    race=grpo["ethnicity"],
                    machine=f"{grpo['manufacturer']} {grpo['scanner model']}",
                    exam=imgs_path,
                    segmentation=[segs_path] if not isna_v2(segs_path) else None,
                    context=context.get_string(),
                    findings=findings.get_string(),
                )
                curr_exams.append(exam)

            for k in slices_without_seg:  # we have an empty segmentation
                exam_id = str(uuid.uuid4())
                contrast_phase = exams_contrast_phase[i]
                imgs_path = self.image_processor.save_image_set(
                    imgs=[processed_slices[k]],
                    exam_id=exam_id + f"_{contrast_phase}_slice_{k}",
                    patient_id=grpo["patient id"],
                )

                if isna_v2(imgs_path):
                    logging.warning(
                        f"Skipping exam {exam_path} for patient {grpo['patient id']} due to processing issues."
                    )
                    continue

                # construct context
                context = JSONReportContext(
                    patient_context=PatientContext(
                        age=bin_age(sanitize_age(grpo["age"])),
                        has_implants=grpo["has implant"],
                    ),
                    exam_context=ExamContext(
                        modality=self._modality,
                        laterality=(
                            BILATERAL if grpo["bilateral mri"] == "yes" else None
                        ),
                        contrast_phase=get_proper_contrast_phase(contrast_phase),
                    ),
                )

                # construct final exam info
                findings = JSONReportFindings(
                    lesion=NOT_PRESENT,
                    assessment=Assessment(
                        birads=get_proper_birads(birads_mapping[1])
                    ),  # negative, since no lesion present is a healthy slice
                )

                exam = ExamInformation(
                    id=exam_id,
                    patient=grpo["patient id"],
                    dataset=grpo["dataset"],
                    modality=self._modality,
                    birads=get_proper_birads(birads_mapping[1]),
                    race=grpo["ethnicity"],
                    machine=f"{grpo['manufacturer']} {grpo['scanner model']}",
                    exam=imgs_path,
                    segmentation=None,
                    context=context.get_string(),
                    findings=findings.get_string(),
                )
                curr_exams.append(exam)

        return curr_exams

    def get_exams_types(self, acquisition_times: list) -> list[str]:
        desc = []

        for i, _ in enumerate(acquisition_times):
            if i == 0:
                curr = "pre-contrast"
            else:
                curr = f"post-contrast phase {i}"

            desc.append(curr)

        return desc
