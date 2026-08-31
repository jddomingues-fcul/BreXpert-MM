import glob
import logging
import os
from argparse import ArgumentParser

import pandas as pd

from dtos.dataset_preprocessing_config import canvas_for
from utils.preprocessing import (
    MODALITIES,
    MODALITY_SHORT,
    birads_category_number,
    isna_v2,
)
from utils.stratified_splitter import (
    load_frozen_assignment,
    save_frozen_assignment,
    save_memmap,
    split_data_stratified,
)

RG = "rg"
FROZEN_SPLITS_FN = "patient_splits.csv"
MEMMAP_COLUMNS = (
    "memmap_file",
    "memmap_index",
    "memmap_shape",
    "mask_memmap_file",
    "has_mask",
)
ALL_MODALITIES = list(MODALITIES)
TRAIN_SPLIT_BIN_EXT = "-train.bin"
VAL_SPLIT_BIN_EXT = "-val.bin"
TEST_SPLIT_BIN_EXT = "-test.bin"

TRAIN_SPLIT_CSV_EXT = "-train.csv"
VAL_SPLIT_CSV_EXT = "-val.csv"
TEST_SPLIT_CSV_EXT = "-test.csv"

BIRADS_GROUPS = {
    0: "additional evaluation (incomplete)",  # removed upstream; here just in case
    1: "healthy (routine)",
    2: "healthy (routine)",
    3: "probably benign (follow up)",
    4: "suspicious/malignancy-likely (biopsy)",
    5: "suspicious/malignancy-likely (biopsy)",
    6: "suspicious/malignancy-likely (biopsy)",  # removed upstream; here just in case
}


def convert_birads(birads: str) -> str:
    category = None if isna_v2(birads) else birads_category_number(str(birads))
    group = BIRADS_GROUPS.get(category)

    if group is None:
        logging.warning(
            f"Unrecognized BI-RADS category: '{birads}'. Passing through as-is."
        )
        return birads

    return group


def load_processed(processed_data_path: str) -> pd.DataFrame:
    parquets = glob.glob(
        os.path.join(processed_data_path, "**/*.parquet"), recursive=True
    )
    parquets = [p for p in parquets if not p.endswith("_build_report.parquet")]
    if parquets:
        logging.info(f"Loading {len(parquets)} processed parquet files")
        return pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)

    csvs = glob.glob(os.path.join(processed_data_path, "**/*.csv"), recursive=True)
    logging.info(f"Loading {len(csvs)} processed csv files")
    return pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)


def prepare_labels(agg_df: pd.DataFrame) -> pd.DataFrame:
    df = agg_df.copy()
    df["collapsed_birads"] = df["birads"].apply(convert_birads)
    crosstab = df.groupby(["collapsed_birads", "birads"]).size()
    logging.info("\n=== BI-RADS COLLAPSE ===\n%s", crosstab)
    return df


def save_splits(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: pd.DataFrame,
        prefix_name: str,
        save_path: str,
        modality: str | None,
) -> dict[str, pd.DataFrame]:
    written: dict[str, pd.DataFrame] = {}
    for name, df, bin_ext, csv_ext in (
            ("train", train_df, TRAIN_SPLIT_BIN_EXT, TRAIN_SPLIT_CSV_EXT),
            ("test", test_df, TEST_SPLIT_BIN_EXT, TEST_SPLIT_CSV_EXT),
            ("val", val_df, VAL_SPLIT_BIN_EXT, VAL_SPLIT_CSV_EXT),
    ):
        if modality is not None:
            bin_path = os.path.join(save_path, f"{prefix_name}{bin_ext}")
            df = save_memmap(
                df=df,
                imgs_filename=bin_path,
                images_shape=canvas_for(modality),
                segs_filename=f"{os.path.splitext(bin_path)[0]}-masks.bin",
            )

        csv_path = os.path.join(save_path, f"{prefix_name}{csv_ext}")
        df.to_csv(csv_path, index=False)
        try:
            df.to_parquet(os.path.splitext(csv_path)[0] + ".parquet", index=False)
        except Exception as exc:
            logging.warning(f"Could not write parquet for {csv_path}: {exc}")
        written[name] = df

    return written


def combine_modalities(
        split_df: pd.DataFrame, packed: list[pd.DataFrame]
) -> pd.DataFrame:
    packed = [df for df in packed if len(df)]
    if not packed:
        return split_df.reindex(columns=[*split_df.columns, *MEMMAP_COLUMNS])

    combined = pd.concat(packed, ignore_index=True)
    leftover = split_df[~split_df["id"].isin(combined["id"])]
    if len(leftover):
        logging.info(
            f"{len(leftover):,} rows are from modalities with no memmap; keeping "
            "them with empty memmap columns."
        )
        leftover = leftover.reindex(columns=combined.columns)
        leftover["has_mask"] = False
        combined = pd.concat([combined, leftover], ignore_index=True)
    return combined


def split_and_save(
        df: pd.DataFrame,
        save_path: str,
        val_frac: float,
        test_frac: float,
        seed: int,
        frozen_path: str | None = None,
) -> None:
    logging.info("Splitting report generation data")
    splits = split_data_stratified(
        df,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=seed,
        frozen=load_frozen_assignment(frozen_path),
    )
    train, val, test = splits.train, splits.val, splits.test

    if frozen_path:
        save_frozen_assignment(frozen_path, splits.assignments)

    packed: dict[str, list[pd.DataFrame]] = {"train": [], "test": [], "val": []}
    for modality in ALL_MODALITIES:
        short = MODALITY_SHORT[modality]
        logging.info(f"Splitting and saving for modality: {modality} ({short})")

        curr_train = train[train["modality"] == modality].copy(deep=True)
        curr_test = test[test["modality"] == modality].copy(deep=True)
        curr_val = val[val["modality"] == modality].copy(deep=True)

        if len(curr_train) == 0 or len(curr_test) == 0 or len(curr_val) == 0:
            logging.warning(f"Skipping modality {modality} due to empty splits")
            continue

        written = save_splits(
            curr_train,
            curr_test,
            curr_val,
            f"{short}-{RG}",
            save_path,
            modality=modality,
        )
        for name, frame in written.items():
            packed[name].append(frame)

    logging.info("Saving splits for all modalities. No modality held out.")
    save_splits(
        combine_modalities(train, packed["train"]),
        combine_modalities(test, packed["test"]),
        combine_modalities(val, packed["val"]),
        f"all-{RG}",
        save_path,
        modality=None,
    )


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--processed_data_path", type=str, required=True)
    args.add_argument("--save_path", type=str, required=True)
    args.add_argument("--val_frac", type=float, default=0.10)
    args.add_argument("--test_frac", type=float, default=0.20)
    args.add_argument("--seed", type=int, default=42)
    args.add_argument(
        "--frozen_splits",
        type=str,
        default=None,
        help=(
            "patient -> split CSV to honour and rewrite. Defaults to "
            f"<save_path>/{FROZEN_SPLITS_FN}, so re-running after adding data keeps "
            "the existing test set intact and only places the new patients."
        ),
    )
    args.add_argument(
        "--reshuffle",
        action="store_true",
        help="Ignore any existing frozen split and draw a fresh one.",
    )
    args.add_argument("--debug", action="store_true", help="Logging debug messages")
    args = args.parse_args()

    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        filename="logs/report_generation_split.log",
        filemode="a",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    os.makedirs(args.save_path, exist_ok=True)

    agg_df = load_processed(args.processed_data_path)
    logging.info("Mapping BI-RADS categories to collapsed groups")
    agg_df = prepare_labels(agg_df)

    frozen_path = args.frozen_splits or os.path.join(args.save_path, FROZEN_SPLITS_FN)
    if args.reshuffle and os.path.exists(frozen_path):
        logging.warning(
            f"--reshuffle: discarding {frozen_path}. Models evaluated on the previous "
            "test set are no longer comparable to ones trained on this split."
        )
        os.remove(frozen_path)

    split_and_save(
        df=agg_df,
        save_path=args.save_path,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
        frozen_path=frozen_path,
    )
