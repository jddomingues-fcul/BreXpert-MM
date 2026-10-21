import os
from argparse import ArgumentParser
from collections.abc import Iterator

import numpy as np
import pandas as pd
from datasets import Dataset, Features, Sequence, Value
from datasets import Image as HFImage
from dotenv import load_dotenv
from PIL import Image

from dtos.breast_cancer_dataset import parse_segmentation_paths
from utils.preprocessing import load_image_frames

FEATURES = Features(
    {
        "image": HFImage(),
        "mask": HFImage(),
        "image_id": Value("string"),
        "source_id": Value("string"),
        "birads": Value("string"),
        # Collapsed group used for stratification, and the v2025 category.
        "collapsed_birads": Value("string"),
        "assessment_scope": Value("string"),
        "modality": Value("string"),
        "dataset": Value("string"),
        "patient": Value("string"),
        "exam_id": Value("string"),
        "breast_id": Value("string"),
        "context": Value("string"),
        "lesion_status": Value("string"),
        "annotation_scope": Value("string"),
        "lesion_count": Value("int32"),
        "n_turns": Value("int32"),
        "conversation": [
            {
                "question": Value("string"),
                "answer": Value("string"),
                "task_type": Value("string"),
                "scope": Value("string"),
                "field_id": Value("string"),
                "difficulty": Value("string"),
                "answer_source": Value("string"),
                "stage": Value("int32"),
                "is_negative": Value("bool"),
                "negative_case": Value("string"),
                "lesion_id": Value("string"),
                "answer_space": Sequence(Value("string")),
                "options_shown": Sequence(Value("string")),
            }
        ],
    }
)

TRAIN_JSONL_PATH = "../data/robust_vqa_split/vqa_train.jsonl"
TRAIN_CSV_PATH = "../data/report_generation_split/all-rg-train.csv"
VAL_JSONL_PATH = "../data/robust_vqa_split/vqa_val.jsonl"
VAL_CSV_PATH = "../data/report_generation_split/all-rg-val.csv"
TEST_JSONL_PATH = "../data/robust_vqa_split/vqa_test.jsonl"
TEST_CSV_PATH = "../data/report_generation_split/all-rg-test.csv"


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Reduce a stored 16-bit frame to the 8 bits a vision tower consumes.

    A straight right-shift, NOT a second min-max stretch. The frame was already
    windowed once when it was read from the source; stretching again here
    re-windows every image independently, which destroys comparability of
    intensities between images -- exactly what an MRI "hyperintense vs not
    hyperintense" or a mammographic mass-density question depends on.
    """
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    if arr.dtype == np.uint16:
        return (arr >> 8).astype(np.uint8)
    # Anything else is already normalised to [0, 1] by the reader.
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _load_mask(paths_cell) -> Image.Image | None:
    """Union of every per-finding mask stored for this image, or None."""
    paths = parse_segmentation_paths(paths_cell)
    if not paths:
        return None

    union = None
    for path in paths:
        try:
            frames = load_image_frames(str(path))
        except Exception:
            continue
        if frames is None:
            continue
        frame = (frames[0] > 0).astype(np.uint8)
        union = frame if union is None else (union | frame)

    if union is None or not union.any():
        return None
    return Image.fromarray((union * 255).astype(np.uint8))


def build_rows(shards: list) -> Iterator[dict]:
    for df in shards:
        for row in df.itertuples():
            img_path = row.image_path
            try:
                frames = load_image_frames(str(img_path))
                pil_img = Image.fromarray(to_uint8(frames[0]))
            except Exception as exc:
                print(f"[skip:load] {row.image_path}: {exc}")
                continue

            conv = row.conversation
            if len(conv) % 2 != 0:
                print(f"[skip:odd] {row.image_path}: {len(conv)} turns")
                continue

            turns = []
            for i in range(0, len(conv), 2):
                q, a = conv[i], conv[i + 1]
                turns.append(
                    {
                        "question": q["text"],
                        "answer": a["text"],
                        "task_type": a["task_type"],
                        "scope": a["scope"],
                        "field_id": a["field_id"],
                        "difficulty": a["difficulty"],
                        "answer_source": a.get("answer_source", "image"),
                        "stage": int(a.get("stage", -1)),
                        "is_negative": bool(a.get("is_negative", False)),
                        "negative_case": a.get("negative_case"),
                        "lesion_id": a.get("lesion_id"),
                        "answer_space": a.get("answer_space") or [],
                        "options_shown": a.get("options_shown") or [],
                    }
                )

            metadata = row.metadata if isinstance(row.metadata, dict) else {}

            yield {
                "image": pil_img,
                "mask": _load_mask(getattr(row, "segmentation", None)),
                "image_id": str(row.image_path),
                "source_id": str(row.source_id),
                "birads": row.birads,
                "collapsed_birads": row.collapsed_birads,
                "assessment_scope": getattr(row, "assessment_scope", None),
                "modality": row.modality,
                "dataset": row.dataset,
                "patient": row.patient,
                "exam_id": getattr(row, "exam_id", None),
                "breast_id": getattr(row, "breast_id", None),
                "context": row.context,
                "lesion_status": metadata.get("lesion_status"),
                "annotation_scope": metadata.get("annotation_scope"),
                "lesion_count": int(metadata.get("lesion_count") or 0),
                "n_turns": len(turns),
                "conversation": turns,
            }


def upload_data_hf(
    df,
    repo_id: str,
    split: str,
    commit_message: str,
    token: str | None = None,
    private: bool = True,
    num_proc: int = 8,
    max_shard_size: str = "500MB",
    dry_run: bool = False,
) -> Dataset:
    token = token or os.environ.get("HF_TOKEN")
    if token is None and not dry_run:
        raise RuntimeError("No HF token: set HF_TOKEN or pass token=")

    shards = [df.iloc[i::64] for i in range(64)]  # 64 shards, 8 workers
    ds = Dataset.from_generator(
        build_rows,
        gen_kwargs={"shards": shards},
        features=FEATURES,
        num_proc=num_proc,
    )

    print(f"{len(ds)} images, {sum(ds['n_turns'])} QA pairs")
    print(f"~{ds.data.nbytes / 1e9:.1f} GB on disk before upload")

    if dry_run:
        return ds

    ds.push_to_hub(
        repo_id=repo_id,
        token=token,
        split=split,
        commit_message=commit_message,
        private=private,
        max_shard_size=max_shard_size,
    )
    return ds


def upload_df(jsonl_path, csv_path, split: str, commit_message: str, token, dry_run):
    df_vqa = pd.read_json(jsonl_path, lines=True)

    parquet_path = os.path.splitext(csv_path)[0] + ".parquet"
    df_csv = (
        pd.read_parquet(parquet_path)
        if os.path.exists(parquet_path)
        else pd.read_csv(csv_path)
    )

    merge_cols = [
        "id",
        "birads",
        "collapsed_birads",
        "modality",
        "dataset",
        "patient",
        "exam_id",
        "breast_id",
        "assessment_scope",
        "segmentation",
    ]
    missing = [c for c in merge_cols if c not in df_csv.columns]
    if missing:
        raise KeyError(
            f"{csv_path} is missing {missing}. Re-run `make report_generation_dataset` "
            "with the current scripts/split.py -- an older split wrote a "
            "`collapsed_birads` column and left `birads` uncollapsed."
        )

    out = df_vqa.merge(
        df_csv[merge_cols].rename(columns={"id": "source_id"}),
        on="source_id",
        how="left",
        indicator="_matched",
    )

    unmatched = int((out["_matched"] == "left_only").sum())
    if unmatched:
        print(f"[warn] {unmatched} dialogues had no matching row in {csv_path}")
    out = out.drop(columns="_matched")

    upload_data_hf(
        out,
        repo_id="jddomingues/brexpert-mm",
        split=split,
        commit_message=commit_message,
        private=True,
        token=token,
        num_proc=8,
        dry_run=dry_run,
    )


SPLITS = {
    "train": (TRAIN_JSONL_PATH, TRAIN_CSV_PATH),
    "val": (VAL_JSONL_PATH, VAL_CSV_PATH),
    "test": (TEST_JSONL_PATH, TEST_CSV_PATH),
}


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument(
        "--split", type=str, choices=["train", "val", "test"], required=True
    )
    args.add_argument(
        "--dry_run",
        action="store_true",
        help="Build the dataset locally and report its size without pushing.",
    )
    args = args.parse_args()

    load_dotenv()
    HF_TOKEN = os.getenv("HF_TOKEN")

    jsonl_path, csv_path = SPLITS[args.split]
    upload_df(
        jsonl_path,
        csv_path,
        split=args.split,
        commit_message=f"Add {args.split} split",
        token=HF_TOKEN,
        dry_run=args.dry_run,
    )
