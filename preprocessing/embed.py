import ast
import logging
import os
import uuid
from functools import partial
from typing import cast

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
    BreastFindings,
    CalcificationAbnormality,
    JSONReportFindings,
    Lesion,
    LesionLocation,
    MassAbnormality,
)
from utils.error_handling import trycatch_func
from utils.image_processor import ImageProcessor
from utils.preprocessing import (
    BIRADS_MAPPING,
    CALCIFICATION,
    CRANIAL_CAUDAL,
    MAMMOGRAPHY,
    MASS,
    MEDIOLATERAL_OBLIQUE,
    NOT_PRESENT,
    UNKNOWN,
    bin_age,
    birads_category_number,
    create_segmentation_mask,
    csv_column_cleaning,
    dview,
    get_proper_birads,
    get_proper_breast_density,
    get_proper_calcification_distribution,
    get_proper_calcification_type,
    get_proper_exam_view,
    get_proper_finding_type,
    get_proper_location,
    get_proper_mass_density,
    get_proper_mass_shape,
    get_value,
    get_value_default,
    isna_v2,
    map_birads,
    migrate_mammography_microlobulated,
    normalize_laterality,
    race_mappings,
    read_breast_image,
    resize_mask,
)

raw_imgs_path = "../data/raw/embed/imgs"
raw_imgs_extension = ".png"
processed_imgs_path = "../data/processed/embed/imgs"
clinical_data_path = "../data/raw/embed/EMBED_OpenData_clinical.csv"
metadata_path = "../data/raw/embed/EMBED_OpenData_metadata_reduced.csv"
clinical_legend_path = "../data/raw/embed/AWS_Open_Data_Clinical_Legend.csv"
imgs_size_path = "../data/raw/embed/image_sizes.csv"
csv_save_path = "../data/processed/embed/embed.csv"


class Embed(BreastCancerDataset):
    _modality = MAMMOGRAPHY

    def __init__(self):
        super().__init__(csv_save_path=csv_save_path)
        self.clinical_df = pd.read_csv(clinical_data_path, low_memory=False)
        self.metadata_df = pd.read_csv(metadata_path, low_memory=False)
        self.clinical_legend_df = pd.read_csv(clinical_legend_path)
        self.imgs_size_df = pd.read_csv(
            imgs_size_path,
            names=[
                "image path",
                "original width",
                "original height",
                "resized width",
                "resized height",
            ],
        )

        self.image_processor = ImageProcessor(
            raw_imgs_path=raw_imgs_path,
            processed_imgs_path=processed_imgs_path,
            image_preprocessing_config=ImagePreprocessingConfig(
                modality=MAMMOGRAPHY,
                read_func=partial(read_breast_image),
            ),
        )

    def process_info(self):
        # ignore cases with spot magnification => we want the entire image
        self.metadata_df = self.metadata_df[self.metadata_df["spot_mag"].isna()]

        # drop the rows whose asses is "0", i.e. "A", or where no assessment was made, i.e. "X"
        self.clinical_df = self.clinical_df[
            (self.clinical_df["asses"] != "A") & (self.clinical_df["asses"] != "X")
            ]

        # converting birads 6 to birads 5
        self.clinical_df["asses"] = self.clinical_df["asses"].replace("K", "M")

        # keeping only interest cols in metadata
        self.metadata_df = self.metadata_df[
            [
                "empi_anon",
                "acc_anon",
                "study_date_anon",
                "FinalImageType",
                "ImageLateralityFinal",
                "ViewPosition",
                "anon_dicom_path",
                "num_roi",
                "ROI_coords",
                "SRC_DST",
                "match_level",
                "Manufacturer",
                "ManufacturerModelName",
                "SeriesDescription",
            ]
        ]

        # keeping only interest cols in clinical
        self.clinical_df = self.clinical_df[
            [
                "massshape",
                "massmargin",
                "massdens",
                "calcfind",
                "calcdistri",
                "calcnumber",
                "otherfind",
                "implanfind",
                "side",
                "location",
                "depth",
                "asses",
                "desc",
                "tissueden",
                "GENDER_DESC",
                "ETHNICITY_DESC",
                "age_at_study",
                "empi_anon",
                "acc_anon",
                "study_date_anon",
                "recc",
                "total_L_find",
                "total_R_find",
            ]
        ]

        # remove no assessed recommendations, and recommendations that say biopsy already performed
        self.clinical_df = self.clinical_df[
            (self.clinical_df["recc"] != "Z") & (self.clinical_df["recc"] != "&")
            ]
        self.clinical_df["recc"] = self.clinical_df["recc"].apply(
            lambda x: self.clean_recc(x) if not isna_v2(x) else x
        )

        # 5. filter for women only for the moment
        self.clinical_df = self.clinical_df[self.clinical_df["GENDER_DESC"] == "Female"]

        # 7. drop duplicates for metadata_df and clinical_df
        self.metadata_df = self.metadata_df.drop_duplicates()
        self.clinical_df = self.clinical_df.drop_duplicates()

        # 7.1 convert dates
        self.metadata_df["study_date_anon"] = pd.to_datetime(
            self.metadata_df["study_date_anon"], errors="coerce"
        )
        self.clinical_df["study_date_anon"] = pd.to_datetime(
            self.clinical_df["study_date_anon"], errors="coerce"
        )

        # 8. Merging metadata and clinical data on exam ID (acc_anon). This will link the clinical data to the file list from metadata
        df_merged = pd.merge(
            self.metadata_df,
            self.clinical_df,
            on=["acc_anon", "empi_anon", "study_date_anon"],
            how="inner",
        )

        # The 'side' column in the clinical data represents the laterality of the finding in that row, and can be
        # L (left), R (right), B (bilateral), or NaN (when there is no finding). Therefore, when merging clinical
        # and metadata, we must first match by exam ID and then match the laterality of the clinical finding (side)
        # to the laterality of the image (ImageLateralityFinal)/ Side "B" and "NaN" can be matched to
        # ImageLateralityFinal both "L" and "R"
        df_merged = df_merged.loc[
            (df_merged.side == df_merged.ImageLateralityFinal)
            | (df_merged.side == "B")
            | (pd.isna(df_merged.side))
            ]

        # 9. Get the column renaming so we can have better descriptions when creating the report
        # construct the legend dictionary to replace the values
        clinical_column_renaming = self.construct_column_renaming()
        legend_dict = self.construct_legend_dict()

        # 10. manual adjustment for tissueden
        df_merged["tissueden"] = df_merged["tissueden"].apply(
            lambda x: (
                legend_dict["tissueden"][str(int(float(x)))] if not isna_v2(x) else None
            )
        )

        # 11. replace the values and then the columns name with cleaning
        # manual adjustments do not get cleaned again
        manual_adjustments = ["tissueden"]

        for header in legend_dict:
            if header in df_merged.columns and header not in manual_adjustments:
                df_merged[header] = df_merged[header].apply(
                    lambda x: self.apply_legend_dict(x, header, legend_dict)
                )

        df_merged = df_merged.rename(columns=clinical_column_renaming)
        df_merged.columns = csv_column_cleaning(list(df_merged.columns))

        # 12. Deal with ethnicity
        ethnicity_edge_cases_dict = {
            "Patient Declines": UNKNOWN,
            "Not Recorded": UNKNOWN,
            "Unknown, Unavailable or Unreported": UNKNOWN,
        }
        df_merged["ethnicity desc"] = df_merged["ethnicity desc"].apply(
            lambda x: ethnicity_edge_cases_dict.get(x, x)
        )

        # 13. Adjust the laterality and view position to be more descriptive
        df_merged["image laterality final"] = df_merged["image laterality final"].apply(
            normalize_laterality
        )
        df_merged["view position"] = df_merged["view position"].apply(
            lambda x: get_value(x, dview) if not isna_v2(x) else None
        )
        df_merged = df_merged[
            df_merged["view position"].isin([MEDIOLATERAL_OBLIQUE, CRANIAL_CAUDAL])
        ].reset_index(
            drop=True
        )  # filter for only the two most common views, to avoid issues with very small number of samples in other views

        # 14. Adjust the age to be integer
        df_merged["age at study"] = df_merged["age at study"].apply(
            lambda x: int(x) if not isna_v2(x) else None
        )

        # 15. Add information about the image size useful for segmentations
        df_merged["image id"] = df_merged["anon dicom path"].apply(
            lambda x: os.path.basename(x)[:-4]
        )
        self.imgs_size_df["image id"] = self.imgs_size_df["image path"].apply(
            lambda x: os.path.basename(x)[:-4]
        )
        out_df = pd.merge(df_merged, self.imgs_size_df, on=["image id"], how="inner")

        # 16. Split DataFrame by patient id, exam id, and image id => this will give each exam for each patient, and each image in the exam
        # each image can have 1+ findings associated with it, so we group and collapse this into one report per image
        # only findings change here
        out_df = out_df.rename(
            columns=lambda c: c.strip().replace(" ", "_").replace("-", "_").lower()
        )

        required_columns = {
            "assessment_bi_rads",
            "calcification_distribution",
            "calcification_finding",
            "image_id",
            "image_laterality_final",
            "location",
            "mass_density",
            "mass_margin",
            "mass_shape",
            "tissue_density",
            "view_position",
        }
        missing = sorted(required_columns - set(out_df.columns))
        if missing:
            raise KeyError(
                f"EMBED: expected columns {missing} after applying the legend "
                f"renaming. The descriptions in {clinical_legend_path} have "
                f"changed upstream -- update the mapping before rebuilding. "
                f"Available columns: {sorted(out_df.columns)}"
            )

        # 17. remove series descriptions to NaN since they do not focus on the breast
        out_df = out_df[out_df["series_description"].notna()]

        # 18. only keeping 2D mammograms, dropping the c-views
        out_df = out_df[out_df["final_image_type"] == "2D"]

        # 18.1 Only keep primary ROI matches. EMBED grades how an ROI was mapped
        # back onto the source mammogram: match_level 1 is a direct primary match
        # and is reliable, level 2 is a secondary match (e.g. a screensave matched
        # to a 2D view, then carried across to the C-view) which the data
        # descriptor explicitly flags as noisier. Since the ROI is what we turn
        # into a segmentation mask, we only trust the primary matches.
        #
        # Both columns are per-ROI *lists* serialised as text ("[1, 2]", "((y,x,y,x),
        # (y,x,y,x))"), so the grading has to be applied element by element -- a
        # scalar `match_level != 1` test is true for every row and would clear
        # every ROI in the collection.
        if "match_level" in out_df.columns:
            before = out_df["roi_coords"].apply(self.count_rois).sum()
            out_df["roi_coords"] = out_df.apply(
                lambda r: self.keep_primary_rois(r["roi_coords"], r["match_level"]),
                axis=1,
            )
            after = out_df["roi_coords"].apply(self.count_rois).sum()
            logging.info(
                f"EMBED: kept {after} of {before} ROIs after dropping non-primary "
                f"(match_level != 1) matches; images without a surviving ROI are still "
                "kept, just without a mask."
            )
        else:
            # No grading column at all: keep every ROI, but still normalise the
            # cell so that "()" becomes None for the emptiness test below.
            logging.warning(
                "EMBED: metadata has no match_level column; ROI provenance cannot be "
                "checked, so secondary matches may end up as masks."
            )
            out_df["roi_coords"] = out_df["roi_coords"].apply(
                lambda v: tuple(self.parse_roi_boxes(v)) or None
            )

        # 18.2 Drop findings we cannot localise at all. Done BEFORE the primary
        # finding is chosen: doing it afterward discarded whole images whose
        # top-ranked row happened to be the unlocalisable one, even when the same
        # image had other perfectly usable findings.
        # `keep_primary_rois` above normalises "no ROI" to None, so isna() is a
        # valid emptiness test here -- the raw column stores the literal "()".
        out_df = out_df[~(out_df["side"].isna() & out_df["roi_coords"].isna())]

        # 19. Keep all findings and select one primary finding per image later.
        # Normalise to the v2025 vocabulary once, here, so the rank below and the
        # BIRADS_MAPPING[1] comparison later both read the same values.
        out_df["assessment_bi_rads"] = out_df["assessment_bi_rads"].apply(
            lambda x: map_birads(x, source="EMBED", modality=MAMMOGRAPHY)
        )
        unmapped = int(out_df["assessment_bi_rads"].isna().sum())
        if unmapped:
            logging.warning(f"EMBED: dropping {unmapped} rows with unmappable BI-RADS.")
        out_df = out_df[out_df["assessment_bi_rads"].notna()].reset_index(drop=True)

        out_df["_birads_rank"] = (
            out_df["assessment_bi_rads"]
            .map(birads_category_number, na_action="ignore")
            .fillna(1)
            .astype(int)
        )
        out_df["_roi_rank"] = out_df["roi_coords"].apply(self.roi_priority)
        out_df["_side_rank"] = out_df["side"].apply(self.side_priority)
        out_df["_lesion_type_rank"] = out_df.apply(self.lesion_type_priority, axis=1)
        out_df["_descriptor_rank"] = out_df.apply(self.descriptor_priority, axis=1)

        out_df = (
            out_df.sort_values(
                by=[
                    "empi_anon",
                    "acc_anon",
                    "image_id",
                    "_birads_rank",
                    "_roi_rank",
                    "_side_rank",
                    "_descriptor_rank",
                    "_lesion_type_rank",
                ],
                ascending=[True, True, True, False, False, False, False, False],
                kind="mergesort",
            )
            .drop(
                columns=[
                    "_birads_rank",
                    "_roi_rank",
                    "_side_rank",
                    "_lesion_type_rank",
                    "_descriptor_rank",
                ]
            )
            .reset_index(drop=True)
        )

        grouped_patients = out_df.groupby(
            by=["empi_anon", "acc_anon", "image_id"], sort=False
        )

        self.report.rows_seen = len(out_df)
        findings_per_image = grouped_patients.size()
        logging.info(
            f"EMBED: {len(out_df)} finding rows across {grouped_patients.ngroups} "
            f"images ({int((findings_per_image > 1).sum())} images carry more than "
            f"one finding, max {int(findings_per_image.max())})."
        )

        for _, g in tqdm(
                grouped_patients,
                total=grouped_patients.ngroups,
                desc="Processing",
                ncols=100,
        ):
            self.append_exams(self.process_patient_exam_img(g))

    @trycatch_func
    def process_patient_exam_img(self, patient_exam_img_df):
        exams = []

        if patient_exam_img_df.empty:
            return exams

        row = patient_exam_img_df.iloc[0]
        exam_id = str(uuid.uuid4())
        race = get_value_default(row["ethnicity_desc"], race_mappings)

        # get the image and the segmentation
        exam_imgs_path = self.image_processor.process_and_save_image(
            os.path.join(
                raw_imgs_path,
                f"{row['image_id']}{raw_imgs_extension}",
            ),
            exam_id,
            str(row["empi_anon"]),
        )

        if isna_v2(exam_imgs_path):
            logging.warning(
                f"Image processing failed for patient {row['empi_anon']}, exam {row['image_id']}. Skipping patient exam image."
            )
            return exams
        exam_imgs_path = str(exam_imgs_path)

        segs_paths = []
        for position, (_, finding_row) in enumerate(patient_exam_img_df.iterrows()):
            segmentation_path = self.save_segmentation_from_rois(
                f"{exam_id}-L{position + 1}",
                str(finding_row.empi_anon),
                finding_row.original_height,
                finding_row.original_width,
                finding_row.resized_height,
                finding_row.resized_width,
                finding_row.roi_coords,
            )
            if segmentation_path is not None:
                segs_paths.append(segmentation_path)

        assessment_birads = get_proper_birads(
            row["assessment_bi_rads"], modality=MAMMOGRAPHY
        )

        context = JSONReportContext(
            patient_context=PatientContext(age=bin_age(row.age_at_study)),
            exam_context=ExamContext(
                modality=self._modality,
                laterality=row.image_laterality_final,
                view=get_proper_exam_view(row.view_position),
            ),
        )

        if row["assessment_bi_rads"] == BIRADS_MAPPING[1]:
            lesions = NOT_PRESENT
        else:
            lesions = [
                self.construct_lesion(finding_row)
                for _, finding_row in patient_exam_img_df.iterrows()
            ]
            lesions = [lesion for lesion in lesions if lesion is not None]
            if lesions:
                lesions[0].is_index = True

        findings = JSONReportFindings.build(
            breast=BreastFindings(
                composition=get_proper_breast_density(row.tissue_density)
            ),
            lesion=lesions,
            assessment=Assessment(birads=assessment_birads),
            annotation_scope=SCOPE_COMPLETE,
        )

        exam = ExamInformation(
            id=exam_id,
            patient=f"{self.get_dataset_name()}-{row.empi_anon}",
            dataset=self.get_dataset_name(),
            modality=self._modality,
            birads=assessment_birads,
            race=race if not isna_v2(race) else UNKNOWN,
            machine=self.build_machine_name(
                row.manufacturer, row.manufacturer_model_name
            ),
            exam=exam_imgs_path,
            segmentation=segs_paths if len(segs_paths) > 0 else None,
            context=context.get_string(),
            findings=findings.get_string(),
            exam_id=f"{self.get_dataset_name()}-{row.acc_anon}",
            breast_id=(
                f"{self.get_dataset_name()}-{row.acc_anon}-"
                f"{row.image_laterality_final}"
            ),
            assessment_scope=ASSESSMENT_GROUNDED,
        )
        exams.append(exam)
        return exams

    def _has_text(self, x):
        return pd.notna(x) and str(x).strip() != ""

    @staticmethod
    def build_machine_name(manufacturer, model) -> str:
        parts = [
            str(part).strip()
            for part in (manufacturer, model)
            if not isna_v2(part) and str(part).strip()
        ]
        return " ".join(parts) if parts else UNKNOWN

    def side_priority(self, side):
        if side in ["left", "right"]:
            return 2
        if side == "both":
            return 1
        return 0

    def descriptor_scores(self, row):
        mass_score = sum(
            [
                self._has_text(row.get("mass_shape")),
                self._has_text(row.get("mass_margin")),
                self._has_text(row.get("mass_density")),
            ]
        )
        calc_score = sum(
            [
                self._has_text(row.get("calcification_finding")),
                self._has_text(row.get("calcification_distribution")),
            ]
        )
        return mass_score, calc_score

    def lesion_type_priority(self, row):
        mass_score, calc_score = self.descriptor_scores(row)
        if mass_score > 0:
            return 2
        if calc_score > 0:
            return 1
        return 0

    def descriptor_priority(self, row):
        mass_score, calc_score = self.descriptor_scores(row)
        return max(mass_score, calc_score)

    @staticmethod
    def parse_roi_boxes(roi_coords) -> list[tuple[int, int, int, int]]:
        """Every well-formed (ymin, xmin, ymax, xmax) box in an ROI_coords cell."""
        if isna_v2(roi_coords):
            return []
        try:
            parsed = (
                roi_coords
                if isinstance(roi_coords, (tuple, list))
                else ast.literal_eval(str(roi_coords))
            )
        except (ValueError, SyntaxError):
            return []

        if not isinstance(parsed, (tuple, list)):
            return []

        boxes = []
        for roi in parsed:
            if not isinstance(roi, (tuple, list)) or len(roi) != 4:
                continue
            try:
                boxes.append(tuple(int(v) for v in roi))
            except (TypeError, ValueError):
                continue
        return boxes  # type: ignore[return-value]

    @staticmethod
    def parse_match_levels(match_level) -> list[int] | None:
        """The per-ROI match grades, or None when the cell says nothing usable."""
        if isna_v2(match_level):
            return None
        if isinstance(match_level, (int, float)) and not isinstance(match_level, bool):
            return [int(match_level)]
        try:
            parsed = (
                match_level
                if isinstance(match_level, (tuple, list))
                else ast.literal_eval(str(match_level))
            )
        except (ValueError, SyntaxError):
            return None

        if isinstance(parsed, (int, float)):
            return [int(parsed)]
        if not isinstance(parsed, (tuple, list)):
            return None
        try:
            return [int(v) for v in parsed]
        except (TypeError, ValueError):
            return None

    @classmethod
    def keep_primary_rois(cls, roi_coords, match_level):
        """Drop every ROI that is not a primary (match_level 1) match.

        Returns a tuple of boxes, or None when nothing usable survives, so that
        `isna` is a valid emptiness test on the resulting column.
        """
        boxes = cls.parse_roi_boxes(roi_coords)
        if not boxes:
            return None

        levels = cls.parse_match_levels(match_level)
        if levels is None:
            # No grading recorded: the ROI provenance is unknown, so it is not a
            # match we said we trust.
            return None

        if len(levels) != len(boxes):
            logging.warning(
                f"EMBED: {len(levels)} match levels for {len(boxes)} ROI boxes; "
                "cannot tell which box is primary, dropping the ROIs for this row."
            )
            return None

        primary = tuple(box for box, level in zip(boxes, levels) if level == 1)
        return primary or None

    @classmethod
    def count_rois(cls, roi_coords) -> int:
        return len(cls.parse_roi_boxes(roi_coords))

    def roi_priority(self, roi_coords):
        return int(self.count_rois(roi_coords) > 0)

    def construct_lesion(self, row: pd.Series) -> Lesion | None:
        if isna_v2(row.get("side")):
            return Lesion()

        has_mass = any(
            [
                self._has_text(row.get("mass_shape")),
                self._has_text(row.get("mass_margin")),
                self._has_text(row.get("mass_density")),
            ]
        )
        has_calc = any(
            [
                self._has_text(row.get("calcification_finding")),
                self._has_text(row.get("calcification_distribution")),
            ]
        )

        location = LesionLocation(
            **get_proper_location(row.location, row.image_laterality_final)
        )

        # EMBED files architectural distortion, asymmetries, dilated ducts and lymph
        # nodes in the `massshape` column. Route those to their own finding type
        # instead of asserting a mass whose every descriptor then drops out.
        shape_value = row.get("mass_shape")
        if self._has_text(shape_value) and get_proper_mass_shape(shape_value) is None:
            routed = get_proper_finding_type(shape_value, MAMMOGRAPHY)
            if routed is not None and routed != MASS:
                return Lesion(location=location, finding_type=routed)

        if has_mass:
            shape, margin = migrate_mammography_microlobulated(
                row.mass_shape, row.mass_margin
            )
            return Lesion(
                location=location,
                finding_type=MASS,
                mass_details=MassAbnormality(
                    shape=shape,
                    margin=margin,
                    density=get_proper_mass_density(row.mass_density),
                ),
            )

        if has_calc:
            calc_type, calc_det = get_proper_calcification_type(
                row.calcification_finding
            )
            return Lesion(
                location=location,
                finding_type=CALCIFICATION,
                calcification_details=CalcificationAbnormality(
                    type=calc_type,
                    type_details=calc_det,
                    distribution=get_proper_calcification_distribution(
                        row.calcification_distribution
                    ),
                ),
            )

        return Lesion(location=location)

    @trycatch_func
    def save_segmentation_from_rois(
            self,
            exam_id,
            patient_id,
            original_height: int,
            original_width: int,
            resized_height: int,
            resized_width: int,
            roi_coords,
    ):

        normalized_rois = tuple(self.parse_roi_boxes(roi_coords))

        if len(normalized_rois) == 0:
            return None

        rois_for_mask = cast(tuple[tuple[int, int, int, int]], normalized_rois)
        mask = create_segmentation_mask(
            height=original_height, width=original_width, rois=rois_for_mask
        )
        # NOTE: Resized because the image itself was resized at download time.
        # resize_mask, not resize_breast_image: a mask must be resampled with
        # nearest neighbour, otherwise interpolated edge values survive the
        # later `> 0` binarisation and grow the annotation.
        mask = resize_mask(mask, (resized_width, resized_height))

        for process in self.image_processor.segmentation_pipeline:
            mask = process(mask)

        if mask.max() == 0:
            logging.info("EMBED: ROI mask is empty after resizing; not saving it.")
            return None

        seg_save_path = os.path.join(
            processed_imgs_path,
            f"{patient_id}-{exam_id}{ImageProcessor.SEGMENTATION_SUFFIX}",
        )
        return self.image_processor.save_process(seg_save_path, [mask])

    @staticmethod
    @trycatch_func
    def apply_legend_dict(val, header, legend_dict):
        assert (
                header in legend_dict.keys()
        ), f"Header {header} not found in the legend dictionary"

        if isna_v2(val):
            return None
        if val == UNKNOWN:
            return UNKNOWN

        res = ""
        vals = [x.strip() for x in str(val).split(",") if x.strip() != ""]
        for item in vals:
            res += f"{legend_dict[header].get(item, '')},"  # In case the value is not found, then we append nothing

        # if the result is empty, i.e "" or ","
        if len(res) <= 1:
            return None

        return res[:-1]

    @trycatch_func
    def construct_column_renaming(self) -> dict:
        clinical_column_renaming = self.clinical_legend_df.drop_duplicates(
            subset=["Header in export", "Discription"]
        )
        clinical_column_renaming = dict(
            clinical_column_renaming[["Header in export", "Discription"]].values
        )

        # Manual adjustments
        clinical_column_renaming["sprocs"] = "Procedure code for the exam"
        clinical_column_renaming["sdate_anon"] = "Unique study identifier"
        clinical_column_renaming["procdate_anon"] = "Procedure Date"
        clinical_column_renaming["pdate_anon"] = "Pathology report date"
        clinical_column_renaming["study_anon"] = "Exam date"
        clinical_column_renaming["stage"] = "TNM Staging"
        clinical_column_renaming["specembed"] = "Specimen embedded"
        clinical_column_renaming["bdistance"] = "Distance in cm (breast)"
        clinical_column_renaming["bdepth"] = "Depth (breast)"
        clinical_column_renaming["loc"] = "Additional location"
        clinical_column_renaming["her2"] = "her2"
        return clinical_column_renaming

    def construct_legend_dict(self) -> dict:
        legend_dict = dict()
        for _, row in self.clinical_legend_df.iterrows():

            if isna_v2(row["Code"]) or isna_v2(row["Meaning"]):
                continue

            header = row["Header in export"]
            code = row["Code"].strip()
            meaning = row["Meaning"].strip()

            # Check if header already exists in the map, if not, create an entry
            if header not in legend_dict:
                legend_dict[header] = dict()

            # Add code and meaning to the corresponding header in the map
            legend_dict[header][code] = meaning.strip().lower()

        # Manual observation adjustment
        for i in range(1, 11):
            legend_dict[f"path{i}"] = legend_dict["path (1-10)"]

        return legend_dict

    @trycatch_func
    def clean_recc(self, recc_str):
        recs = recc_str.split(",")
        len1 = len(recs)
        recs = [rec for rec in recs if rec not in ["Z", "&"]]
        len2 = len(recs)
        if len1 != len2:
            logging.info(f"Cleaned {recc_str} to {','.join(recs)}")
        return ",".join(recs)
