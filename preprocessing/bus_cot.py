import json
import logging
import os
import uuid
from functools import partial

import pandas as pd
from tqdm import tqdm

from dtos.breast_cancer_dataset import (
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
    MassAbnormality,
)
from utils.error_handling import trycatch_func
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    BIRADS_MAPPING,
    MASS,
    NOT_PRESENT,
    PRESENT,
    ULTRASOUND,
    UNKNOWN,
    bin_age,
    collapse_known_biopsy_proven,
    get_proper_birads,
    get_proper_mass_echo_pattern,
    get_proper_mass_margin,
    isna_v2,
    map_birads,
    normalize_laterality,
    read_breast_image,
)

raw_data_path = "../data/raw/BUSCoT/BUS-Expert"
raw_expert_annotated_json_fp = "../data/raw/BUSCoT/DatasetFiles/BUS-Expert_dataset.json"
processed_imgs_path = "../data/processed/BUSCoT/imgs"
csv_save_path = "../data/processed/BUSCoT/BUSCoT.csv"


class BUSCoT(BreastCancerDataset):
    _modality = ULTRASOUND
    _race = UNKNOWN

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_data_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(
                modality=ULTRASOUND, read_func=partial(read_breast_image)
            ),
        )

        # convert the json to a csv
        self.df = self.construct_csv_from_json(raw_expert_annotated_json_fp)

    def process_info(self):
        # 1. filter data to only include rows with valid birads
        self.df = self.df[self.df["birads"].notna()]

        # 2. BUS-CoT records also cover Doppler and elastography. Those are not
        # B-mode ultrasound and must not enter the `us` pool: a colour Doppler
        # frame greyscaled by read_breast_image is not a greyscale sonogram.
        self.df = self.keep_b_mode_only(self.df)

        # 3. map data to correct values
        self.df["birads"] = self.df["birads"].apply(
            lambda x: map_birads(x, source="BUS-CoT", modality=ULTRASOUND)
        )

        unmapped = self.df["birads"].isna().sum()
        if unmapped:
            logging.warning(
                f"Dropping {unmapped} BUS-CoT rows with unmappable BI-RADS."
            )
        self.df = self.df[self.df["birads"].notna()].reset_index(drop=True)

        # 4. BI-RADS 6 is a known-cancer designation, not an imaging read; fold it
        # into 5 as every other adapter does.
        self.df["birads"] = self.df["birads"].apply(collapse_known_biopsy_proven)

        self.run_row_pool(self.df)

    @staticmethod
    def keep_b_mode_only(df):
        non_b_mode = ("doppler", "cdfi", "elasto", "swe", "strain", "contrast")

        normalized = df["modality"].astype(str).str.strip().str.lower()
        drop = normalized.apply(lambda m: any(k in m for k in non_b_mode))

        if drop.any():
            logging.info(
                f"BUS-CoT: dropping {int(drop.sum())} non-B-mode records "
                f"({sorted(set(normalized[drop]))})."
            )

        kept = df[~drop]
        logging.info(
            f"BUS-CoT: modality values kept: {sorted(set(normalized[~drop]))}. "
        )
        return kept.reset_index(drop=True)

    @trycatch_func
    def process_row(self, row):
        exam_id = str(uuid.uuid4())
        source = str(row.dataset).strip().lower().replace(" ", "-")
        patient_id = f"{self.get_dataset_name()}-{source}-{row.patient_id}"
        image_ref = f"{row.pid}-{exam_id}"

        exam_imgs_path = self.image_processor.process_and_save_image(
            os.path.join(raw_data_path, row.img_file_path),
            image_ref,
            patient_id,
        )

        if isna_v2(exam_imgs_path):
            logging.warning(
                f"No images were able to be saved for patient {patient_id}. Skipping this exam."
            )
            return None

        context = JSONReportContext(
            patient_context=PatientContext(age=bin_age(row.age)),
            exam_context=ExamContext(
                modality=self._modality,
                laterality=normalize_laterality(row.side),
            ),
        )

        assessment_birads = get_proper_birads(row.birads, modality=ULTRASOUND)

        findings = JSONReportFindings.build(
            lesion=self.construct_lesions(row),
            assessment=Assessment(birads=assessment_birads),
            annotation_scope=SCOPE_COMPLETE,
        )

        return ExamInformation(
            id=exam_id,
            patient=patient_id,
            dataset=f"{self.get_dataset_name()}/{source}",
            modality=self._modality,
            birads=assessment_birads,
            race=self._race,
            machine=row.machine if not isna_v2(row.machine) else UNKNOWN,
            exam=exam_imgs_path,
            segmentation=None,
            context=context.get_string(),
            findings=findings.get_string(),
            exam_id=f"{patient_id}-{row.pid}",
            breast_id=(
                f"{patient_id}-{normalize_laterality(row.side)}"
                if normalize_laterality(row.side)
                else None
            ),
            assessment_scope=ASSESSMENT_GROUNDED,
        )

    def construct_lesions(self, row):
        if row.birads == BIRADS_MAPPING[1]:
            return NOT_PRESENT

        boundaries = self._as_list(row.mass_margins_a)
        edges = self._as_list(row.mass_margins_b)
        echoes = self._as_list(row.mass_echo)

        count = max(len(boundaries), len(edges), len(echoes))
        if count == 0:
            return PRESENT

        lesions = []
        for i in range(count):
            margin = self.get_custom_mass_margin(
                boundaries[i] if i < len(boundaries) else None,
                edges[i] if i < len(edges) else None,
            )
            echo_pattern = get_proper_mass_echo_pattern(
                echoes[i] if i < len(echoes) else None
            )
            if margin is None and echo_pattern is None:
                lesions.append(Lesion())
            else:
                lesions.append(
                    Lesion(
                        finding_type=MASS,
                        mass_details=MassAbnormality(
                            margin=margin, echo_pattern=echo_pattern
                        ),
                    )
                )

        if lesions:
            lesions[0].is_index = True
        return lesions

    @staticmethod
    def _as_list(value) -> list:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return [v for v in value]
        if isna_v2(value):
            return []
        return [value]

    def get_custom_mass_margin(self, mass_margins_a, mass_margins_b):
        """
        mass_margins_a = LesionBoundary -> demarcation clarity
                        (BoundaryClear > FairlyClear > SomewhatUnclear > Unclear)
        mass_margins_b = LesionEdge     -> contour regularity
                        (Regular > PartiallyRegular > Irregular)

        Rules
        - Poor demarcation is, by definition, "indistinct" -- you can't characterise a contour you can't clearly see.
        - A clearly-seen but non-smooth edge is "not circumscribed"
        - Only a clear boundary AND a regular edge is "circumscribed".
        """
        boundary = (
            None if isna_v2(mass_margins_a) else str(mass_margins_a).strip().lower()
        )
        edge = None if isna_v2(mass_margins_b) else str(mass_margins_b).strip().lower()

        if boundary is None and edge is None:
            return None

        # Poor demarcation dominates regardless of edge.
        if boundary in {"boundarysomewhatunclear", "boundaryunclear"}:
            margin = "indistinct"

        # Boundary is at least fairly clear (or missing) from here on.
        elif edge in {"irregular", "partiallyregular"}:
            # Well enough demarcated to see the contour, but it isn't smooth.
            # Subtype (angular/microlobulated/spiculated) is unknowable -> generic.
            margin = "not circumscribed"

        elif edge == "regular" and boundary in {"boundaryclear", "boundaryfairlyclear"}:
            margin = "circumscribed"

        elif edge == "regular" and boundary is None:
            # Regular edge, no boundary info -> still reads as well-defined.
            margin = "circumscribed"

        elif edge is None and boundary in {"boundaryclear", "boundaryfairlyclear"}:
            # Clear boundary, no edge info -> best guess is circumscribed.
            margin = "circumscribed"

        else:
            return None

        return get_proper_mass_margin(margin, ULTRASOUND)

    @staticmethod
    def construct_csv_from_json(fp):
        with open(fp, "r") as open_file:
            json_content = json.load(open_file)
            pids = []
            datasets = []
            modalities = []
            patient_ids = []
            ages = []
            img_file_paths = []
            machines = []
            sides = []
            birads = []
            mass_margins_a = []
            mass_margins_b = []
            mass_echo = []

            for pid in tqdm(
                    json_content,
                    desc="Processing JSON",
                    unit="pid",
                    ncols=100,
                    total=len(json_content),
            ):
                pids.append(pid)
                datasets.append(json_content[pid]["dataset"])
                modalities.append(
                    None
                    if not json_content[pid]["modality"]
                    else json_content[pid]["modality"]
                )
                patient_ids.append(json_content[pid]["patient_id"])
                ages.append(json_content[pid]["age"])
                img_file_paths.append(json_content[pid]["image_file"]["raw_image"])
                machines.append(
                    None
                    if not json_content[pid]["device_type"]
                    else json_content[pid]["device_type"]
                )
                sides.append(
                    None if not json_content[pid]["side"] else json_content[pid]["side"]
                )

                us_report = json_content[pid].get("us_report") or {}

                ordered = sorted(
                    us_report.values(),
                    key=lambda v: not bool(v.get("IsMainLesion", False)),
                )

                lead = ordered[0] if ordered else {}
                birads.append(lead.get("BIRADS", None))
                mass_margins_a.append([v.get("LesionBoundary") for v in ordered])
                mass_margins_b.append([v.get("LesionEdge") for v in ordered])
                mass_echo.append([v.get("EchoCharacteristics") for v in ordered])

        df = pd.DataFrame(
            {
                "pid": pids,
                "dataset": datasets,
                "modality": modalities,
                "patient_id": patient_ids,
                "age": ages,
                "img_file_path": img_file_paths,
                "machine": machines,
                "side": sides,
                "birads": birads,
                "mass_margins_a": mass_margins_a,
                "mass_margins_b": mass_margins_b,
                "mass_echo": mass_echo,
            }
        )
        return df
