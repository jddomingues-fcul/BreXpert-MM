import hashlib
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm

from dtos.breast_cancer_dataset import parse_segmentation_paths
from utils.preprocessing import birads_category_number, isna_v2, load_image_frames

logger = logging.getLogger(__name__)


def assert_patient_disjoint(
        train_patients: set, val_patients: set, test_patients: set
) -> None:
    logger.info("\n=== LEAKAGE CHECK ===")
    logger.info(f"Train ∩ Val:  {len(train_patients & val_patients)}")
    logger.info(f"Train ∩ Test: {len(train_patients & test_patients)}")
    logger.info(f"Val ∩ Test:   {len(val_patients & test_patients)}")

    overlaps = {
        "train/val": train_patients & val_patients,
        "train/test": train_patients & test_patients,
        "val/test": val_patients & test_patients,
    }
    leaking = {name: ids for name, ids in overlaps.items() if ids}
    if leaking:
        raise AssertionError(
            "Patient leakage between splits: "
            + "; ".join(f"{name}: {len(ids)} patients" for name, ids in leaking.items())
        )


def _has_any_segmentation(df: pd.DataFrame) -> bool:
    if "segmentation" not in df.columns:
        return False
    return any(parse_segmentation_paths(value) for value in df["segmentation"])


def save_memmap(
        df: pd.DataFrame,
        imgs_filename: str,
        images_shape: tuple,
        segs_filename: str | None = None,
) -> pd.DataFrame:
    if len(df) == 0:
        logger.warning("Dataframe is empty, no data to save")
        return df

    width, height = images_shape
    n_images = len(df)
    imgs_arr = np.memmap(
        imgs_filename, dtype=np.uint16, mode="w+", shape=(n_images, height, width)
    )

    # Only pay for a mask file when the split actually carries segmentations; a
    # blank one still costs n_images * height * width bytes on disk.
    segs_arr = None
    write_masks = segs_filename is not None and _has_any_segmentation(df)
    if write_masks:
        segs_arr = np.memmap(
            segs_filename, dtype=np.uint8, mode="w+", shape=(n_images, height, width)
        )
    elif segs_filename is not None:
        logger.info(
            f"No segmentations in this split; skipping mask memmap {segs_filename}"
        )

    failed_indices = []
    memmap_indices: list[int] = []
    has_mask: list[bool] = []
    multiframe = 0
    index = 0
    for row_idx, row in enumerate(
            tqdm(
                df.itertuples(),
                total=n_images,
                desc="Saving images",
                unit="image",
                leave=False,
            )
    ):
        try:
            frames = load_image_frames(row.exam)
            if frames is None:
                raise ValueError(f"Could not load {row.exam}")
            if len(frames) > 1:
                multiframe += 1
            imgs_arr[index] = frames[0]

            wrote_mask = False
            if segs_arr is not None:
                paths = parse_segmentation_paths(getattr(row, "segmentation", None))
                if paths:
                    union = np.zeros((height, width), dtype=np.uint8)
                    for path in paths:
                        mask = load_image_frames(path)
                        if mask is None:
                            continue
                        union |= (mask[0] > 0).astype(np.uint8)
                    segs_arr[index] = union
                    wrote_mask = bool(union.any())

            memmap_indices.append(index)
            has_mask.append(wrote_mask)
            index += 1
        except Exception as e:
            failed_indices.append(row_idx)
            logger.error(f"Error saving image {row.exam}: {e}")

    if multiframe:
        logger.warning(
            f"{multiframe} of {n_images} exams have more than one frame; only the "
            "first frame is written to the memmap."
        )

    if failed_indices:
        logger.warning(
            f"{len(failed_indices)} images failed to load. Truncating memmap from {n_images} to {index} entries."
        )
        imgs_arr.flush()
        del imgs_arr
        os.truncate(
            imgs_filename, index * height * width * np.dtype(np.uint16).itemsize
        )

        if segs_arr is not None:
            segs_arr.flush()
            del segs_arr
            os.truncate(
                str(segs_filename), index * height * width * np.dtype(np.uint8).itemsize
            )

        df.drop(df.index[failed_indices], inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        imgs_arr.flush()
        if segs_arr is not None:
            segs_arr.flush()

    # Add memmap metadata to the dataframe
    df["memmap_file"] = os.path.basename(imgs_filename)
    df["memmap_index"] = memmap_indices
    df["memmap_shape"] = [f"{height}x{width}"] * len(memmap_indices)
    df["mask_memmap_file"] = (
        os.path.basename(str(segs_filename)) if write_masks else None
    )
    df["has_mask"] = has_mask
    return df


TRAIN, VAL, TEST = "train", "val", "test"
SPLITS = (TRAIN, VAL, TEST)


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    assignments: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def patients(self) -> dict[str, set]:
        return {
            TRAIN: set(self.train["patient"]),
            VAL: set(self.val["patient"]),
            TEST: set(self.test["patient"]),
        }


def load_frozen_assignment(path: str | None) -> dict[str, str]:
    """Read a previously written patient -> split artifact."""
    if not path or not os.path.exists(path):
        return {}
    frozen = pd.read_csv(path)
    missing = [c for c in ("patient", "split") if c not in frozen.columns]
    if missing:
        raise KeyError(f"Frozen split file {path} is missing columns {missing}")
    bad = set(frozen["split"]) - set(SPLITS)
    if bad:
        raise ValueError(f"Frozen split file {path} has unknown splits {sorted(bad)}")
    assignment = dict(zip(frozen["patient"].astype(str), frozen["split"]))
    logger.info(f"Loaded {len(assignment):,} frozen patient assignments from {path}")
    return assignment


def save_frozen_assignment(path: str, assignments: pd.DataFrame) -> None:
    assignments.to_csv(path, index=False)
    logger.info(f"Wrote {len(assignments):,} patient assignments to {path}")


def assign_patient_strata(
        agg_df: pd.DataFrame,
        label_col: str = "collapsed_birads",
        modality_col: str = "modality",
        severity_col: str | None = "birads",
        dataset_col: str | None = "dataset",
        min_stratum_patients: int = 3,
) -> pd.DataFrame:
    needed = [c for c in ("patient", modality_col, label_col) if c is not None]
    df = agg_df.loc[:, [c for c in needed if c in agg_df.columns]].copy()

    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise KeyError(
            f"Cannot stratify: {missing} not in the dataframe. Run "
            "scripts.split.prepare_labels first so `collapsed_birads` exists."
        )

    use_dataset = bool(dataset_col) and dataset_col in agg_df.columns
    if use_dataset:
        df[dataset_col] = agg_df[dataset_col]
    elif dataset_col:
        logger.warning(
            f"No `{dataset_col}` column; stratifying on ({modality_col}, {label_col}) "
            "only. Small sources may end up entirely in one split."
        )

    if severity_col and severity_col in agg_df.columns:
        df["_severity"] = (
            agg_df[severity_col]
            .map(lambda v: birads_category_number(str(v)) if not isna_v2(v) else None)
            .fillna(-1)
            .astype(int)
        )
    else:
        df["_severity"] = 0

    unlabelled = df[label_col].isna() | df["patient"].isna()
    if unlabelled.any():
        logger.warning(
            f"{int(unlabelled.sum())} rows have no patient or no {label_col}; "
            "they are assigned to train and excluded from stratification."
        )
        df = df[~unlabelled]

    # Primary modality: most images, alphabetical tie-break.
    primary_modality = _primary_by_count(df, modality_col)

    # Most severe label, most-frequent label as tie-break.
    label_counts = (
        df.groupby(["patient", label_col])
        .agg(n=("_severity", "size"), severity=("_severity", "max"))
        .reset_index()
    )
    label_counts = label_counts.sort_values(
        ["patient", "severity", "n", label_col],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    primary_label = label_counts.drop_duplicates("patient", keep="first")[
        ["patient", label_col]
    ]

    rows_per_patient = df.groupby("patient").size().rename("n_rows").reset_index()

    strata = primary_modality.merge(primary_label, on="patient").merge(
        rows_per_patient, on="patient"
    )
    if use_dataset:
        strata = strata.merge(_primary_by_count(df, dataset_col), on="patient")

    keys = [strata[modality_col].astype(str), strata[label_col].astype(str)]
    if use_dataset:
        keys.insert(0, strata[dataset_col].astype(str))
    strata["stratum"] = _coarsen(keys, min_stratum_patients)
    return strata


def _primary_by_count(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Per patient, the most frequent value of `column` (alphabetical tie-break)."""
    counts = df.groupby(["patient", column]).size().rename("n").reset_index()
    counts = counts.sort_values(
        ["patient", "n", column], ascending=[True, False, True], kind="mergesort"
    )
    return counts.drop_duplicates("patient", keep="first")[["patient", column]]


def _coarsen(keys: list[pd.Series], min_patients: int) -> pd.Series:
    strata = None
    for start in range(len(keys)):
        candidate = keys[start:]
        label = (
            candidate[0].str.cat(candidate[1:], sep=" | ")
            if len(candidate) > 1
            else candidate[0].copy()
        )
        if strata is None:
            strata = label
            continue
        too_small = strata.groupby(strata).transform("size") < min_patients
        if not too_small.any():
            break
        strata = strata.where(~too_small, label)
    return strata


def _allocate(
        patients: list[str],
        weights: dict[str, int],
        val_frac: float,
        test_frac: float,
        budget_by: str,
        preassigned: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    preassigned = preassigned or {}
    fracs = {
        TRAIN: 1.0 - val_frac - test_frac,
        VAL: val_frac,
        TEST: test_frac,
    }
    sizes = {p: (weights[p] if budget_by == "rows" else 1) for p in patients}
    total_rows = sum(sizes.values())

    assigned: dict[str, list[str]] = {TRAIN: [], VAL: [], TEST: []}
    counts = {split: 0 for split in SPLITS}
    filled = {split: 0.0 for split in SPLITS}

    # Largest first; `patients` arrives in stable hash order, which breaks ties
    # between patients of equal size.
    order = {patient: i for i, patient in enumerate(patients)}
    for placed, patient in enumerate(
            sorted(patients, key=lambda p: (-sizes[p], order[p]))
    ):
        split = preassigned.get(patient)
        if split is None:
            split = max(
                SPLITS,
                key=lambda s: (
                    round(fracs[s] * (placed + 1) - counts[s], 9),
                    round(total_rows * fracs[s] - filled[s], 9),
                    fracs[s],
                    s,
                ),
            )
            assigned[split].append(patient)
        else:
            assigned[split].append(patient)
        counts[split] += 1
        filled[split] += sizes[patient]

    return assigned


def _enforce_minimum(
        assigned: dict[str, list[str]],
        min_per_split: int,
        weights: dict[str, int],
        locked: set[str],
) -> None:
    if min_per_split <= 0:
        return
    if sum(len(v) for v in assigned.values()) < 3 * min_per_split:
        return

    def movable(split: str) -> list[str]:
        # Smallest first, so the row budgets shift as little as possible.
        return sorted(
            (p for p in assigned[split] if p not in locked),
            key=lambda p: (weights.get(p, 1), p),
        )

    for split in (TEST, VAL):
        while len(assigned[split]) < min_per_split:
            donors = [
                d
                for d in SPLITS
                if d != split and len(assigned[d]) > min_per_split and movable(d)
            ]
            if not donors:
                break
            donor = max(
                donors, key=lambda d: (len(assigned[d]) - min_per_split, d == TRAIN)
            )
            patient = movable(donor)[0]
            assigned[donor].remove(patient)
            assigned[split].append(patient)


def split_data_stratified(
        agg_df: pd.DataFrame,
        val_frac: float = 0.10,
        test_frac: float = 0.20,
        seed: int = 42,
        label_col: str = "collapsed_birads",
        modality_col: str = "modality",
        severity_col: str | None = "birads",
        dataset_col: str | None = "dataset",
        min_per_split: int = 1,
        budget_by: str = "rows",
        frozen: dict[str, str] | None = None,
) -> SplitResult:
    if not 0 <= val_frac < 1 or not 0 <= test_frac < 1 or val_frac + test_frac >= 1:
        raise ValueError(
            f"val_frac + test_frac must leave room for train; got "
            f"val={val_frac}, test={test_frac}"
        )
    if budget_by not in ("rows", "patients"):
        raise ValueError(f"budget_by must be 'rows' or 'patients', got {budget_by!r}")

    strata = assign_patient_strata(
        agg_df,
        label_col=label_col,
        modality_col=modality_col,
        severity_col=severity_col,
        dataset_col=dataset_col,
        min_stratum_patients=3 * max(min_per_split, 1),
    )
    weights = dict(zip(strata["patient"], strata["n_rows"]))

    frozen = frozen or {}
    known = set(strata["patient"])
    reused = sum(1 for patient in frozen if patient in known)
    if frozen:
        logger.info(
            f"Honouring {reused:,} frozen assignments; "
            f"{len(known) - reused:,} patients are new."
        )

    assignment: dict[str, str] = {}
    stratum_of: dict[str, str] = {}
    coverage_failures: list[str] = []

    logger.info("=== STRATA (patients / images) ===")
    for stratum, group in sorted(
            strata.groupby("stratum"), key=lambda item: str(item[0])
    ):
        patients = sorted(
            group["patient"],
            key=lambda p: hashlib.blake2b(
                f"{seed}|{p}".encode(), digest_size=8
            ).hexdigest(),
        )

        assigned = _allocate(
            patients, weights, val_frac, test_frac, budget_by, preassigned=frozen
        )
        _enforce_minimum(assigned, min_per_split, weights, locked=set(frozen))

        for split, members in assigned.items():
            for patient in members:
                assignment[patient] = split
                stratum_of[patient] = stratum

        short = len(patients) >= 3 * max(min_per_split, 1) and not (
                assigned[VAL] and assigned[TEST]
        )
        if short and any(p in frozen for p in patients):
            logger.warning(
                f"Stratum {stratum} is missing from val or test, but its patients "
                "are frozen; leaving it as the frozen file dictates."
            )
        elif short:
            coverage_failures.append(stratum)

        logger.info(
            f"{stratum!s:70s}: {len(patients):>6,} patients / "
            f"{sum(weights[p] for p in patients):>7,} images -> "
            f"train {len(assigned[TRAIN]):>5,} | val {len(assigned[VAL]):>4,} | "
            f"test {len(assigned[TEST]):>4,}"
        )

    if coverage_failures:
        raise AssertionError(
            "Strata large enough to populate three splits are missing from val "
            "or test: " + "; ".join(coverage_failures)
        )

    split_series = agg_df["patient"].map(lambda p: assignment.get(p, TRAIN))
    unstratified = agg_df.loc[~agg_df["patient"].isin(assignment), "patient"]
    if len(unstratified):
        logger.warning(
            f"{unstratified.nunique():,} patients ({len(unstratified):,} rows) could "
            "not be stratified (no patient id or no label) and default to train."
        )

    result = SplitResult(
        train=agg_df[split_series == TRAIN].copy(),
        val=agg_df[split_series == VAL].copy(),
        test=agg_df[split_series == TEST].copy(),
        assignments=pd.DataFrame(
            {
                "patient": list(assignment),
                "split": [assignment[p] for p in assignment],
                "stratum": [stratum_of[p] for p in assignment],
                "seed": seed,
            }
        ).sort_values("patient", ignore_index=True),
    )

    patients = result.patients
    assert_patient_disjoint(patients[TRAIN], patients[VAL], patients[TEST])
    _log_distribution(agg_df, split_series, label_col, modality_col, dataset_col)
    return result


def _log_distribution(
        agg_df, split_series, label_col, modality_col, dataset_col=None
) -> None:
    labelled = agg_df.assign(split=split_series)

    totals = labelled.groupby("split").agg(
        images=("id", "count"), patients=("patient", "nunique")
    )
    totals["image_share"] = (totals["images"] / totals["images"].sum()).round(3)
    totals["patient_share"] = (totals["patients"] / totals["patients"].sum()).round(3)
    totals["images_per_patient"] = (totals["images"] / totals["patients"]).round(2)
    logger.info("\n=== SPLIT TOTALS ===\n%s", totals)

    for name, column in (("PATIENT", "patient"), ("IMAGE", "id")):
        aggregate = "nunique" if column == "patient" else "count"
        dist = (
            labelled.groupby([modality_col, label_col, "split"])[column]
            .agg(aggregate)
            .unstack(fill_value=0)
            .sort_index()
        )
        logger.info(f"\n=== {name} DISTRIBUTION ===\n%s", dist)

    if dataset_col and dataset_col in labelled.columns:
        by_source = (
            labelled.groupby([dataset_col, "split"])["patient"]
            .nunique()
            .unstack(fill_value=0)
            .sort_index()
        )
        logger.info("\n=== PATIENTS PER SOURCE ===\n%s", by_source)
