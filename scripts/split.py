import glob
import logging
import os
from argparse import ArgumentParser

import pandas as pd

from dtos.dataset_preprocessing_config import RESIZE_DIMS
from utils.stratified_splitter import save_memmap, split_data_budgeted

RG = "rg"
ALL_MODALITIES = ["mg", "mr", "us"]
TRAIN_SPLIT_BIN_EXT = "-train.bin"
VAL_SPLIT_BIN_EXT = "-val.bin"
TEST_SPLIT_BIN_EXT = "-test.bin"

TRAIN_SPLIT_CSV_EXT = "-train.csv"
VAL_SPLIT_CSV_EXT = "-val.csv"
TEST_SPLIT_CSV_EXT = "-test.csv"


def convert_birads(birads: str) -> str:
    if birads in ["(1) negative", "(2) benign"]:
        return "healthy/routine"

    elif birads == "(3) probably benign":
        return "probably benign (follow up)"

    elif birads in ["(4) suspicious", "(5) highly suggestive of malignancy"]:
        return "suspicious/malignancy-likely (biopsy)"

    # This will never happen because we removed these cases, but we have it here just in case
    elif birads == "(0) additional evaluation":
        return "additional evaluation (incomplete)"

    elif birads == "(6) known biopsy proven":
        return "known biopsy proven (malignancy)"

    else:
        logging.warning(
            f"Unrecognized BI-RADS category: '{birads}'. Passing through as-is."
        )
        return birads


def save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: pd.DataFrame,
    prefix_name: str,
    save_path: str,
) -> None:
    save_memmap(
        df=train_df,
        imgs_filename=os.path.join(save_path, f"{prefix_name}{TRAIN_SPLIT_BIN_EXT}"),
        images_shape=RESIZE_DIMS,
    )
    train_df.to_csv(
        os.path.join(save_path, f"{prefix_name}{TRAIN_SPLIT_CSV_EXT}"), index=False
    )

    save_memmap(
        df=test_df,
        imgs_filename=os.path.join(save_path, f"{prefix_name}{TEST_SPLIT_BIN_EXT}"),
        images_shape=RESIZE_DIMS,
    )
    test_df.to_csv(
        os.path.join(save_path, f"{prefix_name}{TEST_SPLIT_CSV_EXT}"), index=False
    )

    save_memmap(
        df=val_df,
        imgs_filename=os.path.join(save_path, f"{prefix_name}{VAL_SPLIT_BIN_EXT}"),
        images_shape=RESIZE_DIMS,
    )
    val_df.to_csv(
        os.path.join(save_path, f"{prefix_name}{VAL_SPLIT_CSV_EXT}"), index=False
    )


def split_and_save(
    df: pd.DataFrame,
    save_path: str,
) -> None:
    # Split
    logging.info("Splitting report generation data")
    train, test, val = split_data_budgeted(df)

    # save all modalities together
    logging.info("Saving splits for all modalities. No modality held out.")
    prefix_name = f"all-{RG}"
    save_splits(train, test, val, prefix_name, save_path)

    # Save per modality
    for modality in ALL_MODALITIES:
        logging.info(f"Splitting and saving for modality: {modality}")

        curr_train = train[train["modality"] == modality].copy(deep=True)
        curr_test = test[test["modality"] == modality].copy(deep=True)
        curr_val = val[val["modality"] == modality].copy(deep=True)

        if len(curr_train) == 0 or len(curr_test) == 0 or len(curr_val) == 0:
            logging.warning(f"Skipping modality {modality} due to empty splits")
            continue

        prefix_name = f"{modality}-{RG}"
        save_splits(curr_train, curr_test, curr_val, prefix_name, save_path)


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--processed_data_path", type=str, required=True)
    args.add_argument("--save_path", type=str, required=True)
    args.add_argument("--debug", action="store_true", help="Logging debug messages")
    args = args.parse_args()

    logging.basicConfig(
        filename="logs/report_generation_split.log",
        filemode="a",
        format="%(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    # Create the save path if it does not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # get all the csv files in the processed data path
    csvs = glob.glob(os.path.join(args.processed_data_path, "**/*.csv"), recursive=True)

    # Load all the csv dataframes and concat them all together, ignoring the index
    logging.info("Loading all the csv files")
    agg_df = pd.concat([pd.read_csv(csv) for csv in csvs], ignore_index=True)

    # Map birads to new categories
    logging.info("Mapping BI-RADS categories to new categories")
    agg_df["original_birads"] = agg_df["birads"]
    agg_df["birads"] = agg_df["birads"].apply(convert_birads)

    split_and_save(
        df=agg_df,
        save_path=args.save_path,
    )
