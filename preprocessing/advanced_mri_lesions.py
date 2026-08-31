import glob
import logging
import math
import os
import random
import uuid
from collections import defaultdict
from copy import deepcopy

import numpy as np
import pandas as pd
import pydicom as dicom

from dtos.breast_cancer_dataset import (
    ASSESSMENT_DERIVED,
    ASSESSMENT_GROUNDED,
    BreastCancerDataset,
    ExamInformation,
)
from dtos.dataset_preprocessing_config import ImagePreprocessingConfig
from dtos.json_report_context import ExamContext, JSONReportContext, PatientContext
from dtos.json_report_findings import (
    SCOPE_COMPLETE,
    Assessment,
    JSONReportFindings,
    Lesion,
    LesionLocation,
)
from utils.error_handling import trycatch_func
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    BILATERAL,
    BIRADS_MAPPING,
    LATERALITY_MAP,
    MASS,
    MRI,
    NOT_PRESENT,
    UNKNOWN,
    bin_age,
    collapse_known_biopsy_proven,
    column_cleaning_csv_reading,
    csv_column_cleaning,
    get_or_default,
    get_proper_birads,
    get_proper_contrast_phase,
    get_value,
    isna_v2,
    map_birads,
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
    _machines = "1.5 T"
    _race = UNKNOWN

    _MIN_SLICE_DISTANCE_FOR_NEGATIVES = 6

    # Negatives emitted per exam per side (before / after the lesion).
    _NEGATIVES_PER_SIDE = 1

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(modality=MRI),
        )
        self.mri_lesions_df = pd.read_excel(mri_lesions_path, sheet_name=0, header=1)
        self.metadata_df = column_cleaning_csv_reading(metadata_path).reset_index()

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
        if "file size" not in adjusted_columns:
            raise ValueError(
                "Expected a 'file size' column in the Advanced-MRI metadata.csv to "
                "re-seat the shifted headers. The upstream layout has changed -- "
                "verify the column mapping before rebuilding."
            )
        adjusted_columns.insert(adjusted_columns.index("file size"), UNKNOWN)
        if len(adjusted_columns) != len(self.metadata_df.columns):
            raise ValueError(
                f"Advanced-MRI metadata header re-seating produced {len(adjusted_columns)} "
                f"names for {len(self.metadata_df.columns)} columns."
            )
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

        # 8. Create the segmentation paths in the segmentation metadata. A subject
        # whose SEG folder is missing or empty simply has no mask; it must not take
        # the whole build down.
        seg_metadata_df["segmentation path"] = seg_metadata_df["file location"].apply(
            self.resolve_segmentation_path
        )
        missing = int(seg_metadata_df["segmentation path"].isna().sum())
        if missing:
            logging.warning(
                f"Advanced-MRI: {missing} of {len(seg_metadata_df)} SEG entries have no "
                "file on disk; those subjects will be processed without a mask."
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
        self.mri_lesions_df["segmentation path"] = self.mri_lesions_df.apply(
            lambda x: self.remove_non_mri_segmentation_paths(x), axis=1
        )  # type: ignore

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
            self.mri_lesions_df["birads"] != BIRADS_MAPPING[0]
            ].reset_index(drop=True)

        # 15. swap known biopsy proven to highly suggestive of malignancy
        self.mri_lesions_df["birads"] = self.mri_lesions_df["birads"].apply(
            collapse_known_biopsy_proven
        )

        self.run_row_pool(self.mri_lesions_df)

    @trycatch_func
    def process_row(self, row):
        folder_path = row["file location"][2:]  # to remove the "./" from the beginning
        pat = f"{self.get_dataset_name()}-{row['subject id']}"
        seg_path = row["segmentation path"]
        resulting_exams = []

        searchable_folder = os.path.join(
            self.image_processor.raw_imgs_path, folder_path
        )
        instances = self.collect_series_instances(
            searchable_folder, row.get("series description", "")
        )
        if not instances:
            logging.warning(
                f"No usable axial DICOM instances under {searchable_folder}."
            )
            return resulting_exams

        ordered_locations = sorted({inst["location"] for inst in instances})
        location_rank = {loc: rank for rank, loc in enumerate(ordered_locations)}

        if isna_v2(seg_path) or not os.path.exists(seg_path):
            logging.warning(
                f"Segmentation path does not exist for patient {pat} on the images at path {folder_path}. Saving images only."
            )
            seg_data = None
        else:
            seg_data = dicom.dcmread(seg_path)

        # Resolve every annotated lesion in this exam to a slice position first.
        # Negatives are then chosen relative to *all* of them, so one lesion's
        # "healthy" slice can no longer land on top of another lesion.
        lesion_ranks = {}
        for i in range(1, 7):
            slice_coord = row[f"slice coord lesion{i}"]
            if isna_v2(slice_coord) or row[f"type lesion{i}"] is None:
                continue

            matched = self.find_matching_location(ordered_locations, slice_coord)
            if matched is None:
                logging.warning(
                    f"No slice matching coordinate {slice_coord} for patient {pat}."
                )
                continue
            lesion_ranks[i] = (location_rank[matched], matched, slice_coord)

        if not lesion_ranks:
            return resulting_exams

        resulting_exams.extend(
            self._positive_exams(row, pat, instances, lesion_ranks, seg_data)
        )
        resulting_exams.extend(
            self._negative_exams(row, pat, instances, ordered_locations, lesion_ranks)
        )
        return resulting_exams

    def _positive_exams(self, row, pat, instances, lesion_ranks, seg_data):
        lesions_by_location: dict[float, list[tuple[int, object]]] = defaultdict(list)
        for i, (_, matched_location, slice_coord) in lesion_ranks.items():
            lesions_by_location[matched_location].append((i, slice_coord))

        exams = []
        for matched_location, entries in lesions_by_location.items():
            lesions = self._lesions_at(row, entries)
            segs = self._segs_at(seg_data, entries)

            for inst in instances:
                if inst["location"] != matched_location:
                    continue
                exam = self.build_exam(
                    row=row,
                    pat=pat,
                    instance=inst,
                    lesion=lesions,
                    birads=row["birads"],
                    segs=segs,
                    assessment_scope=ASSESSMENT_GROUNDED,
                )
                if exam is not None:
                    exams.append(exam)
        return exams

    def _lesions_at(self, row, entries):
        if row["birads"] == BIRADS_MAPPING[1]:
            return NOT_PRESENT

        lesions = [
            lesion
            for lesion in (self._construct_lesion(row, i) for i, _ in entries)
            if lesion is not None
        ]
        if lesions:
            lesions[0].is_index = True
        return lesions

    def _segs_at(self, seg_data, entries):
        if seg_data is None:
            return []
        segs = []
        for _, slice_coord in entries:
            seg = self.get_corresponding_seg(seg_data, slice_coord)
            if seg is not None:
                segs.append(seg)
        return segs

    def _negative_exams(self, row, pat, instances, ordered_locations, lesion_ranks):
        exams = []
        for negative_location in self.pick_negative_locations(
                ordered_locations, {rank for rank, _, _ in lesion_ranks.values()}, pat
        ):
            for inst in instances:
                if inst["location"] != negative_location:
                    continue
                exam = self.build_exam(
                    row=row,
                    pat=pat,
                    instance=inst,
                    lesion=NOT_PRESENT,
                    birads=BIRADS_MAPPING[1],
                    segs=[],
                    assessment_scope=ASSESSMENT_DERIVED,
                )
                if exam is not None:
                    exams.append(exam)
                break  # one contrast phase is enough for a negative
        return exams

    def build_exam(self, row, pat, instance, lesion, birads, segs, assessment_scope):
        exam_id = str(uuid.uuid4())

        curr_img = self.image_processor.process_image(instance["path"])
        if curr_img is None:
            logging.warning(
                f"Image processing failed for patient {pat} on the image at path {instance['path']}. Skipping exam."
            )
            return None

        sois_path = self.image_processor.save_image_set([curr_img], exam_id, pat)
        if isna_v2(sois_path):
            return None

        segs_paths = []
        for position, seg in enumerate(segs or []):
            saved = self.image_processor.save_segmentation_set(
                [seg], f"{exam_id}-L{position + 1}", pat
            )
            if not isna_v2(saved):
                segs_paths.append(saved)

        context = JSONReportContext(
            patient_context=PatientContext(
                age=bin_age(row["age at mri"]),
                has_implants=get_or_default(row["breast implants"], None),
            ),
            exam_context=ExamContext(
                modality=MRI,
                laterality=BILATERAL,
                contrast_phase=get_proper_contrast_phase(instance["contrast_phase"]),
            ),
        )

        assessment_birads = (
            get_proper_birads(birads, modality=MRI) if birads is not None else None
        )

        findings = JSONReportFindings.build(
            lesion=lesion,
            assessment=(
                Assessment(birads=assessment_birads)
                if assessment_birads is not None
                else None
            ),
            annotation_scope=SCOPE_COMPLETE,
        )

        manufacturer = get_or_default(row["manufacturer"], "")
        machine = f"{manufacturer} {self._machines}".strip()

        return ExamInformation(
            id=exam_id,
            patient=pat,
            dataset=self.get_dataset_name(),
            modality=MRI,
            birads=assessment_birads if assessment_birads is not None else UNKNOWN,
            race=self._race,
            machine=machine if machine else UNKNOWN,
            exam=sois_path,
            segmentation=segs_paths or None,
            context=context.get_string(),
            findings=findings.get_string(),
            exam_id=pat,
            breast_id=None,  # bilateral acquisition
            assessment_scope=assessment_scope,
        )

    @staticmethod
    def slice_orientation(ds) -> str | None:
        """axial / sagittal / coronal, from the direction cosines."""
        iop = getattr(ds, "ImageOrientationPatient", None)
        if iop is None or len(iop) != 6:
            return None
        try:
            vectors = np.asarray([float(v) for v in iop], dtype=float).reshape(2, 3)
        except (TypeError, ValueError):
            return None
        normal = np.cross(vectors[0], vectors[1])
        return ("sagittal", "coronal", "axial")[int(np.argmax(np.abs(normal)))]

    def collect_series_instances(
            self, folder: str, series_description: str = ""
    ) -> list[dict]:
        entries = []
        for path in sorted(glob.glob(os.path.join(folder, "*"))):
            if os.path.isdir(path) or not os.path.exists(path):
                continue
            try:
                ds = dicom.dcmread(path, stop_before_pixels=True)
            except Exception as e:
                logging.warning(f"Could not read DICOM header at {path}: {e}")
                continue

            location = getattr(ds, "SliceLocation", None)
            if location is None:
                continue

            # The lesion coordinates in the clinical sheet are axial slice
            # positions. A sagittal series numbers its slices along a different
            # patient axis, so a numeric match there is a coincidence, not the
            # same anatomy.
            orientation = self.slice_orientation(ds)
            if orientation is not None and orientation != "axial":
                continue

            # PALETTE COLOR series (the TRAM parametric maps) store colour-table
            # indices; read as greyscale they are meaningless, and they are
            # derived maps rather than acquisitions in any case.
            photometric = (
                str(getattr(ds, "PhotometricInterpretation", "")).strip().upper()
            )
            if photometric.startswith("PALETTE"):
                continue

            entries.append(
                {
                    "path": path,
                    "location": round(float(location), 2),
                    "instance_number": int(getattr(ds, "InstanceNumber", 0) or 0),
                    "temporal": getattr(ds, "TemporalPositionIdentifier", None),
                }
            )

        self.assign_contrast_phases(entries, series_description)
        return entries

    @staticmethod
    def assign_contrast_phases(entries: list[dict], series_description: str) -> None:
        desc = str(series_description).strip().lower()

        if "multiphase" not in desc.replace(" ", ""):
            fixed = None
            if "t2" in desc:
                fixed = "t2"
            elif "t1" in desc:
                fixed = "t1"
            elif "mask" in desc:
                # The VIBRANT "MASK" acquisition is the pre-contrast reference.
                fixed = "pre-contrast"
            for entry in entries:
                entry["contrast_phase"] = fixed
            return

        # Within one slice position, a multiphase series differs only by phase.
        by_location: dict[float, list[dict]] = {}
        for entry in entries:
            by_location.setdefault(entry["location"], []).append(entry)

        for group in by_location.values():
            group.sort(
                key=lambda e: (
                    int(e["temporal"])
                    if e["temporal"] is not None
                    else e["instance_number"]
                )
            )
            for phase_idx, entry in enumerate(group):
                entry["contrast_phase"] = (
                    "pre-contrast"
                    if phase_idx == 0
                    else f"post-contrast phase {phase_idx}"
                )

    def find_matching_location(self, ordered_locations, pos_str, tol=1e-2):
        try:
            target = float(pos_str)
        except (TypeError, ValueError):
            return None

        for location in ordered_locations:
            if math.isclose(location, target, abs_tol=tol):
                return location
        return None

    def pick_negative_locations(self, ordered_locations, lesion_ranks, seed_key):
        if not lesion_ranks:
            return []

        rng = random.Random(f"advanced-mri-negatives-{seed_key}")
        min_distance = self._MIN_SLICE_DISTANCE_FOR_NEGATIVES

        def far_enough(rank):
            return all(abs(rank - lesion) >= min_distance for lesion in lesion_ranks)

        before = [r for r in range(min(lesion_ranks)) if far_enough(r)]
        after = [
            r
            for r in range(max(lesion_ranks) + 1, len(ordered_locations))
            if far_enough(r)
        ]

        picked = []
        for candidates in (before, after):
            if not candidates:
                continue
            picked.extend(
                rng.sample(candidates, min(self._NEGATIVES_PER_SIDE, len(candidates)))
            )

        return [ordered_locations[rank] for rank in picked]

    def _construct_lesion(self, row, i) -> Lesion | None:
        if isna_v2(row[f"laterality lesion{i}"]):
            return None
        return Lesion(
            location=LesionLocation(laterality=row[f"laterality lesion{i}"]),
            finding_type=MASS,
        )

    @staticmethod
    def resolve_segmentation_path(file_location) -> str | None:
        if isna_v2(file_location):
            return None
        matches = sorted(
            glob.glob(os.path.join(raw_imgs_path, str(file_location)[2:], "*"))
        )
        return matches[0] if matches else None

    @staticmethod
    def remove_non_mri_segmentation_paths(row):
        # NOTE: Only these samples actually match on the segmentation when manually validating metadata so these are the ones we use
        desc = str(row.get("series description", "")).strip().lower()
        if desc != "Registered AX Sen Vibrant MultiPhase".lower():
            return None
        return row["segmentation path"]

    def get_corresponding_seg(self, seg_data, slice):
        try:
            target = float(slice)
        except (TypeError, ValueError):
            return None

        seg_index = -1
        for i, gs in enumerate(seg_data["PerFrameFunctionalGroupsSequence"].value):
            frame_coord = float(
                gs["PlanePositionSequence"].value[0]["ImagePositionPatient"].value[-1]
            )
            # Compare numerically. The previous substring match on the rounded
            # string made "-45.5" match "-45.55" and "145.5" alike.
            if math.isclose(frame_coord, target, abs_tol=1e-2):
                seg_index = i
                break

        if seg_index == -1:
            return None

        if seg_index >= len(seg_data.pixel_array):
            return None

        seg_slice = seg_data.pixel_array[seg_index]

        if seg_slice.max() == seg_slice.min():
            return None

        return self.image_processor.apply_processing(seg_slice, is_segmentation=True)

    def fill_in_contents(self):
        # breast implants info
        self.mri_lesions_df["breast implants"] = self.mri_lesions_df[
            "breast implants"
        ].apply(lambda x: get_value(x, yes_no_mapping))

        # birads info
        self.mri_lesions_df["birads"] = self.mri_lesions_df["birads"].apply(
            lambda x: map_birads(x, source="Advanced-MRI-Breast-Lesions", modality=MRI)
        )

        # positions for tumors
        for i in range(1, 7):
            self.mri_lesions_df[f"laterality lesion{i}"] = self.mri_lesions_df[
                f"pos{i}"
            ].apply(
                lambda x: (
                    get_value(str(x)[0], LATERALITY_MAP) if not isna_v2(x) else None
                )
            )
            self.mri_lesions_df[f"slice coord lesion{i}"] = self.mri_lesions_df[
                f"pos{i}"
            ].apply(lambda x: get_pos_coord_slice(str(x)) if not isna_v2(x) else None)
        self.mri_lesions_df = self.mri_lesions_df.drop(
            [f"pos{i}" for i in range(1, 7)], axis=1
        )


def get_pos_coord_slice(x) -> str | None:
    if x is None:
        return None

    if isinstance(x, str):
        return x[1:]
    return None


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
