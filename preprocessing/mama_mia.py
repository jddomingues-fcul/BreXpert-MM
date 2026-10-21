import ast
import logging
import os
from functools import partial

import numpy as np
import pandas as pd

from dtos.breast_cancer_dataset import (
    ASSESSMENT_DERIVED,
    ASSESSMENT_GROUNDED,
    BreastCancerDataset,
    ExamInformation,
    stable_exam_id,
)
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext, PatientContext
from dtos.json_report_findings import (
    SCOPE_COMPLETE,
    SCOPE_INDEX_ONLY,
    Assessment,
    BreastFindings,
    JSONReportFindings,
)
from utils.error_handling import pool_worker
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    BILATERAL,
    BIRADS_MAPPING,
    MRI,
    NOT_PRESENT,
    PRESENT,
    UNKNOWN,
    bin_age,
    csv_column_cleaning,
    get_proper_birads,
    get_proper_contrast_phase,
    get_proper_mri_fgt,
    get_value,
    isna_v2,
    normalize_to_uint16,
    race_mappings,
    read_nii_gz_images,
    read_nii_gz_raw,
    yes_no_mapping,
)

raw_imgs_path = "../data/raw/mama-mia/images"
raw_segs_path = "../data/raw/mama-mia/segmentations/expert"
raw_imgs_extension = ".nii.gz"
processed_imgs_path = "../data/processed/mama-mia/imgs"
clinical_data_path = "../data/raw/mama-mia/clinical_and_imaging_info.xlsx"
csv_save_path = "../data/processed/mama-mia/mama-mia.csv"


class MamaMia(BreastCancerDataset):
    _modality = MRI
    _birads = "highly suggestive of malignancy"  # all patients have cancer (birads 6), but we set this to the highest non-biopsy proven value (birads 5)

    _SLICES_AFTER_AND_BEFORE_LESION_FOR_NEGATIVES = 4
    _NEGATIVES_PER_SIDE = 1

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(
                modality=MRI, read_func=partial(read_nii_gz_images)
            ),
        )

        self.clinical_df = pd.read_excel(clinical_data_path)
        self.clinical_df.columns = csv_column_cleaning(list(self.clinical_df.columns))

    def process_info(self):
        self.report.rows_seen = len(self.clinical_df)
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

        self.run_pool(
            self.process_patient, grouped_patients, "Processing mama-mia patients"
        )

    @pool_worker
    def process_patient(self, grpo):
        curr_exams = []

        if len(grpo) > 1:
            logging.warning(
                f"Patient {grpo['patient id'].iloc[0]} has more than one row in the clinical data. Only the first row will be used."
            )

        grpo = grpo.iloc[
            0
        ]  # NOTE: The dataframe per patient will always only 1 row, so we access it this way

        patient_id = f"{self.get_dataset_name()}-{grpo['patient id']}"

        available_imgs = sorted(
            os.listdir(os.path.join(raw_imgs_path, grpo["patient id"]))
        )
        available_imgs = [
            path for path in available_imgs if path.endswith(raw_imgs_extension)
        ]

        exams_contrast_phase = self.get_contrast_phases(grpo, len(available_imgs))

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
            return curr_exams

        # DCE-MRI phases only mean something relative to each other: the lesion is
        # what brightens between pre- and post-contrast. Normalising each phase
        # against its own min/max would rescale that difference away, so all the
        # phases of a patient are normalised against shared statistics.
        volume_range = self.get_shared_intensity_range(grpo, available_imgs)

        # Slices covered by the expert segmentation.
        lesion_slices = [
            idx for idx in range(seg_slices.shape[0]) if np.sum(seg_slices[idx]) > 0
        ]
        if not lesion_slices:
            logging.warning(
                f"No non-empty segmentation slice for patient {grpo['patient id']}. Skipping."
            )
            return curr_exams

        # Keep the central slice only => avoid near-duplicate slices of one lesion.
        central_lesion_slice = lesion_slices[len(lesion_slices) // 2]

        # Lesion-free negatives are measured from the FULL lesion extent, not from the central slice
        slices_without_seg = self.get_lesion_free_slices(
            grpo, lesion_slices, seg_slices.shape[0]
        )

        for i, exam_path in enumerate(available_imgs):
            exam_path = os.path.join(raw_imgs_path, grpo["patient id"], exam_path)

            exam_slices = read_nii_gz_raw(exam_path)
            if isna_v2(exam_slices):
                logging.warning(f"Exam images of {exam_path} were not able to be read")
                continue

            exam_slices = normalize_to_uint16(exam_slices, value_range=volume_range)

            if exam_slices.shape[0] != seg_slices.shape[0]:
                logging.warning(
                    f"Slice count mismatch for patient {grpo['patient id']} on {exam_path}: "
                    f"{exam_slices.shape[0]} image slices vs {seg_slices.shape[0]} segmentation slices. Skipping."
                )
                continue

            processed_slices = np.array(
                [self.image_processor.apply_processing(slice) for slice in exam_slices]
            )

            contrast_phase = exams_contrast_phase[i]

            exam = self.build_exam(
                grpo=grpo,
                patient_id=patient_id,
                img=processed_slices[central_lesion_slice],
                seg=seg_slices[central_lesion_slice],
                slice_idx=central_lesion_slice,
                contrast_phase=contrast_phase,
                lesion=PRESENT,
                birads=self._birads,
            )
            if exam is not None:
                curr_exams.append(exam)

            for k in slices_without_seg:
                exam = self.build_exam(
                    grpo=grpo,
                    patient_id=patient_id,
                    img=processed_slices[k],
                    seg=None,
                    slice_idx=k,
                    contrast_phase=contrast_phase,
                    lesion=NOT_PRESENT,
                    birads=BIRADS_MAPPING[1],  # no lesion on this slice
                )
                if exam is not None:
                    curr_exams.append(exam)

        return curr_exams

    def build_exam(
        self,
        grpo,
        patient_id: str,
        img,
        seg,
        slice_idx: int,
        contrast_phase,
        lesion,
        birads: str | None,
    ):
        exam_id = stable_exam_id(
            self.get_dataset_name(), grpo["patient id"], contrast_phase, slice_idx
        )
        slice_ref = f"{exam_id}_{contrast_phase}_slice_{slice_idx}"

        imgs_path = self.image_processor.save_image_set(
            imgs=[img], exam_id=slice_ref, patient_id=patient_id
        )
        if isna_v2(imgs_path):
            logging.warning(
                f"Skipping slice {slice_idx} for patient {patient_id} due to processing issues."
            )
            return None

        segs_path = (
            self.image_processor.save_segmentation_set(
                segs=[seg], exam_id=slice_ref, patient_id=patient_id
            )
            if seg is not None
            else None
        )

        context = JSONReportContext(
            patient_context=PatientContext(
                age=bin_age(grpo["age"]),
                has_implants=grpo["has implant"],
            ),
            exam_context=ExamContext(
                modality=self._modality,
                laterality=(BILATERAL if grpo["bilateral mri"] == "yes" else None),
                contrast_phase=get_proper_contrast_phase(contrast_phase),
            ),
        )

        assessment_birads = (
            get_proper_birads(birads, modality=MRI) if birads is not None else None
        )

        multifocal = (
            grpo.get("multifocal cancer") == "yes"
            or grpo.get("bilateral breast cancer") == "yes"
        )

        findings = JSONReportFindings.build(
            breast=BreastFindings(fgt=get_proper_mri_fgt(grpo["breast density"])),
            lesion=lesion,
            assessment=(
                Assessment(birads=assessment_birads)
                if assessment_birads is not None
                else None
            ),
            annotation_scope=SCOPE_INDEX_ONLY if multifocal else SCOPE_COMPLETE,
        )

        machine = " ".join(
            str(v)
            for v in (grpo["manufacturer"], grpo["scanner model"])
            if not isna_v2(v)
        )

        return ExamInformation(
            id=exam_id,
            patient=patient_id,
            dataset=f"{self.get_dataset_name()}/{str(grpo['dataset']).strip().lower()}",
            modality=self._modality,
            birads=assessment_birads if assessment_birads is not None else UNKNOWN,
            race=grpo["ethnicity"],
            machine=machine if machine else UNKNOWN,
            exam=imgs_path,
            segmentation=[segs_path] if not isna_v2(segs_path) else None,
            context=context.get_string(),
            findings=findings.get_string(),
            exam_id=patient_id,
            breast_id=None,  # bilateral acquisition
            assessment_scope=(
                ASSESSMENT_GROUNDED if seg is not None else ASSESSMENT_DERIVED
            ),
        )

    def get_lesion_free_slices(
        self, grpo, lesion_slices: list[int], n_slices: int
    ) -> list[int]:
        if grpo.get("multifocal cancer") == "yes" or (
            grpo.get("bilateral breast cancer") == "yes"
        ):
            return []

        buffer = self._SLICES_AFTER_AND_BEFORE_LESION_FOR_NEGATIVES
        first_lesion_slice, last_lesion_slice = min(lesion_slices), max(lesion_slices)

        before_end = first_lesion_slice - buffer
        before = list(range(max(0, before_end - self._NEGATIVES_PER_SIDE), before_end))

        after_start = last_lesion_slice + buffer + 1
        after = list(
            range(after_start, min(n_slices, after_start + self._NEGATIVES_PER_SIDE))
        )

        lesion_slice_set = set(lesion_slices)
        return [
            k for k in before + after if 0 <= k < n_slices and k not in lesion_slice_set
        ]

    def get_shared_intensity_range(self, grpo, available_imgs: list[str]):
        """Min/max across every contrast phase of one patient."""
        mins, maxs = [], []
        for rel_path in available_imgs:
            volume = read_nii_gz_raw(
                os.path.join(raw_imgs_path, grpo["patient id"], rel_path)
            )
            if isna_v2(volume):
                continue
            mins.append(float(np.nanmin(volume)))
            maxs.append(float(np.nanmax(volume)))

        if not mins or not maxs:
            return None
        return min(mins), max(maxs)

    def get_contrast_phases(self, grpo, n_images: int) -> list:
        phases: list = []
        if not isna_v2(grpo["acquisition times"]):
            try:
                phases = self.get_exams_types(
                    ast.literal_eval(str(grpo["acquisition times"]))
                )
            except (ValueError, SyntaxError) as e:
                logging.warning(
                    f"Could not parse acquisition times for patient {grpo['patient id']}: {e}"
                )

        if len(phases) != n_images:
            if phases:
                logging.warning(
                    f"Patient {grpo['patient id']}: {len(phases)} acquisition times for "
                    f"{n_images} volumes. Falling back to positional phase naming."
                )
            phases = self.get_exams_types(range(n_images))

        return phases

    def get_exams_types(self, acquisition_times) -> list[str]:
        desc = []

        for i, _ in enumerate(acquisition_times):
            if i == 0:
                curr = "pre-contrast"
            else:
                curr = f"post-contrast phase {i}"

            desc.append(curr)

        return desc
