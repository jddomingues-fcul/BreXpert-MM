import logging
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.preprocessing import isna_v2, load_images_from_npy


def save_memmap(
    df: pd.DataFrame,
    imgs_filename: str,
    images_shape: tuple,
    segs_filename: Optional[str] = None,
) -> None:
    # NOTE: The df here is expected to have the same columns as the original dataframe: dtos/breast_cancer_dataset.py
    if len(df) == 0:
        logging.warning("Dataframe is empty, no data to save")
        return

    n_images = len(df)
    imgs_arr = np.memmap(
        imgs_filename,
        dtype=np.uint16,
        mode="w+",
        shape=(n_images, images_shape[0], images_shape[1]),
    )

    segs_arr = None
    if segs_filename is not None and not isna_v2(segs_filename):
        segs_arr = np.memmap(
            segs_filename,
            dtype=np.uint16,
            mode="w+",
            shape=(n_images, images_shape[0], images_shape[1]),
        )  # type: ignore

    failed_indices = []
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
            img = load_images_from_npy(row.exam)
            imgs_arr[index] = img

            if segs_arr is not None:
                seg = load_images_from_npy(row.segmentation)
                segs_arr[index] = seg

            index += 1
        except Exception as e:
            failed_indices.append(row_idx)
            logging.error(f"Error saving image {row.exam}: {e}")

    if failed_indices:
        logging.warning(
            f"{len(failed_indices)} images failed to load. Truncating memmap from {n_images} to {index} entries."
        )
        imgs_arr.flush()
        # Rewrite memmap with correct size
        truncated = np.memmap(
            imgs_filename,
            dtype=np.uint16,
            mode="r+",
            shape=(index, images_shape[0], images_shape[1]),
        )
        truncated.flush()
        del truncated

        if segs_arr is not None:
            segs_arr.flush()
            truncated_seg = np.memmap(
                segs_filename,
                dtype=np.uint16,
                mode="r+",
                shape=(index, images_shape[0], images_shape[1]),
            )
            truncated_seg.flush()
            del truncated_seg

        # Also drop the failed rows from the dataframe so CSV matches memmap
        df.drop(df.index[failed_indices], inplace=True)
        df.reset_index(drop=True, inplace=True)
    else:
        imgs_arr.flush()
        if segs_arr is not None:
            segs_arr.flush()


def split_data_budgeted(
    agg_df: pd.DataFrame,
    val_total: int = 200,
    test_total: int = 400,
    min_val_quotas: dict[tuple[str, str], int] | None = None,
    min_test_quotas: dict[tuple[str, str], int] | None = None,
    heavy_patient_train_bias: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    df = agg_df[["patient", "modality", "birads"]].dropna().drop_duplicates().copy()

    # Rows per patient in full dataset
    patient_row_count = agg_df.groupby("patient")["id"].count().to_dict()

    # Patient -> set[(modality, birads)]
    patient_labels = (
        df.groupby("patient")
        .apply(lambda x: set((m, b) for m, b in zip(x["modality"], x["birads"])))
        .to_dict()
    )

    all_patients = sorted(patient_labels.keys())
    total_patients = len(all_patients)

    # Label -> number of unique patients
    label_counts = Counter()
    for labels in patient_labels.values():
        label_counts.update(labels)

    logging.info("=== PATIENT COUNTS PER LABEL ===")
    for (modality, birads), count in sorted(label_counts.items()):
        logging.info(f"{modality:2s} | {birads:45s}: {count:,}")

    # Default quotas
    if min_val_quotas is None:
        min_val_quotas = {
            ("us", "healthy/routine"): 40,
            ("us", "probably benign (follow up)"): 40,
            ("us", "suspicious/malignancy-likely (biopsy)"): 60,
            ("mr", "healthy/routine"): 20,
            ("mr", "suspicious/malignancy-likely (biopsy)"): 20,
            ("mg", "healthy/routine"): 20,
            ("mg", "probably benign (follow up)"): 20,
            ("mg", "suspicious/malignancy-likely (biopsy)"): 20,
        }

    if min_test_quotas is None:
        min_test_quotas = {
            ("us", "healthy/routine"): 75,
            ("us", "probably benign (follow up)"): 75,
            ("us", "suspicious/malignancy-likely (biopsy)"): 125,
            ("mr", "healthy/routine"): 40,
            ("mr", "suspicious/malignancy-likely (biopsy)"): 40,
            ("mg", "healthy/routine"): 40,
            ("mg", "probably benign (follow up)"): 40,
            ("mg", "suspicious/malignancy-likely (biopsy)"): 40,
        }

    # Approximate targets for all labels
    def proportional_target(label: tuple[str, str], split_total: int) -> int:
        n = label_counts[label]
        if n < 10:
            return 0
        return int(round(n * split_total / total_patients))

    desired_val = {
        label: proportional_target(label, val_total) for label in label_counts
    }
    desired_test = {
        label: proportional_target(label, test_total) for label in label_counts
    }

    # Enforce minimum quotas
    for label, q in min_val_quotas.items():
        if label in label_counts:
            desired_val[label] = max(desired_val.get(label, 0), q)

    for label, q in min_test_quotas.items():
        if label in label_counts:
            desired_test[label] = max(desired_test.get(label, 0), q)

    # Prevent impossible over-allocation
    for label, total_n in label_counts.items():
        # leave at least 1 patient in train when possible
        max_holdout = max(0, total_n - 1) if total_n > 1 else 0
        requested = desired_val.get(label, 0) + desired_test.get(label, 0)
        if requested > max_holdout and requested > 0:
            scale = max_holdout / requested
            desired_val[label] = int(np.floor(desired_val[label] * scale))
            desired_test[label] = int(np.floor(desired_test[label] * scale))

    unassigned = set(all_patients)
    val_patients: set[str] = set()
    test_patients: set[str] = set()
    val_counts = Counter()
    test_counts = Counter()

    def update_counts(counter: Counter, patient: str) -> None:
        for label in patient_labels[patient]:
            counter[label] += 1

    def deficit_score(
        patient: str, current: Counter, desired: dict[tuple[str, str], int]
    ) -> float:
        score = 0.0  # score because a patient might help mutliple labels, and so has a higher score
        for label in patient_labels[patient]:
            deficit = desired.get(label, 0) - current.get(label, 0)
            if deficit > 0:
                # rare labels are more valuable
                # if a label is rare, this gives the score an additional edge to rank higher
                score += 1.0 + (5.0 / max(1, label_counts[label]))

        if heavy_patient_train_bias:
            # in case of a tie, we prefer patients with less amount of data
            # log1p does log(1+x) => counts start at 0/1 and grow
            # 0.0005 arbitrary small value
            score -= 0.0005 * np.log1p(patient_row_count.get(patient, 0))

        return score

    def build_split(
        name: str,
        split_set: set[str],
        split_counts: Counter,
        desired: dict[tuple[str, str], int],
        split_total: int,
    ) -> None:
        logging.info(f"\n=== BUILDING {name.upper()} ===")
        while len(split_set) < split_total:
            candidates = list(unassigned)
            if not candidates:
                raise ValueError(
                    f"Not enough patients to meet desired quotas for split (only {len(split_set)} / {split_total})"
                )

            scores = [(deficit_score(p, split_counts, desired), p) for p in candidates]
            best_score, best_patient = max(scores, key=lambda x: x[0])

            if best_score <= 0:
                raise ValueError(
                    f"Not enough patients to meet desired quotas for split (only {len(split_set)} / {split_total})"
                )

            split_set.add(best_patient)
            unassigned.remove(best_patient)
            update_counts(split_counts, best_patient)

        logging.info(f"{name.upper()} final: {len(split_set):,} patients")

    # Build test first, then val
    build_split("test", test_patients, test_counts, desired_test, test_total)
    build_split("val", val_patients, val_counts, desired_val, val_total)

    train_patients = set(unassigned)

    train_df = agg_df[agg_df["patient"].isin(train_patients)].copy()
    val_df = agg_df[agg_df["patient"].isin(val_patients)].copy()
    test_df = agg_df[agg_df["patient"].isin(test_patients)].copy()

    # Verification
    logging.info("\n=== LEAKAGE CHECK ===")
    logging.info(f"Train ∩ Val:  {len(train_patients & val_patients)}")
    logging.info(f"Train ∩ Test: {len(train_patients & test_patients)}")
    logging.info(f"Val ∩ Test:   {len(val_patients & test_patients)}")

    split_df = df.copy()
    split_df["split"] = split_df["patient"].map(
        lambda p: (
            "train" if p in train_patients else ("val" if p in val_patients else "test")
        )
    )

    patient_dist = (
        split_df.groupby(["modality", "birads", "split"])["patient"]
        .nunique()
        .unstack(fill_value=0)
        .sort_index()
    )
    logging.info("\n=== PATIENT DISTRIBUTION ===")
    logging.info("\n%s", patient_dist)

    row_dist = (
        agg_df.assign(
            split=agg_df["patient"].map(
                lambda p: (
                    "train"
                    if p in train_patients
                    else ("val" if p in val_patients else "test")
                )
            )
        )
        .groupby(["modality", "birads", "split"])["id"]
        .count()
        .unstack(fill_value=0)
        .sort_index()
    )
    logging.info("\n=== ROW DISTRIBUTION ===")
    logging.info("\n%s", row_dist)

    return train_df, test_df, val_df
