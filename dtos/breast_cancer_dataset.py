from __future__ import annotations

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import asdict, dataclass, field
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.error_handling import pool_worker
from utils.preprocessing import (
    drain_unmapped_counts,
    format_unmapped_report,
    merge_unmapped_counts,
)

logger = logging.getLogger(__name__)

ASSESSMENT_GROUNDED = "grounded"
ASSESSMENT_EXAM_LEVEL = "exam_level"
ASSESSMENT_DERIVED = "derived"
ASSESSMENT_SCOPES = (ASSESSMENT_GROUNDED, ASSESSMENT_EXAM_LEVEL, ASSESSMENT_DERIVED)


def _worker_count() -> int:
    """One core left for the parent, but never fewer than one worker."""
    return max(1, cpu_count() - 1)


def stable_exam_id(dataset: str, *source_parts) -> str:
    return str(
        uuid.uuid5(uuid.NAMESPACE_URL, json.dumps([dataset, *map(str, source_parts)]))
    )


@dataclass(frozen=True)
class ExamInformation:
    id: str
    patient: str
    dataset: str
    modality: str
    birads: str
    race: str
    machine: str

    exam: str  # path for the exam
    segmentation: list[str] | None  # path for the segmentations (if any)
    context: str
    findings: str

    exam_id: str | None = None
    breast_id: str | None = None
    assessment_scope: str = ASSESSMENT_EXAM_LEVEL

    def __post_init__(self):
        if self.assessment_scope not in ASSESSMENT_SCOPES:
            raise ValueError(
                f"assessment_scope must be one of {list(ASSESSMENT_SCOPES)}, "
                f"got {self.assessment_scope!r}"
            )
        if self.exam_id is None:
            object.__setattr__(self, "exam_id", self.id)

    def to_row(self) -> dict:
        row = asdict(self)
        row["segmentation"] = (
            json.dumps(self.segmentation) if self.segmentation else None
        )
        row["n_segmentations"] = len(self.segmentation) if self.segmentation else 0
        return row


def parse_segmentation_paths(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"Could not parse segmentation cell {text!r}; treating as none.")
        return []
    if isinstance(parsed, str):
        return [parsed]
    return [str(v) for v in parsed] if isinstance(parsed, list) else []


@dataclass
class BuildReport:
    dataset: str = ""
    rows_seen: int = 0
    filtered_rows: int = 0
    exams_kept: int = 0
    drops: Counter = field(default_factory=Counter)
    unmapped: Counter = field(default_factory=Counter)

    def record_drop(self, reason: str, count: int = 1) -> None:
        self.drops[reason] += count

    def merge(self, other: BuildReport | None) -> None:
        if other is None:
            return
        self.rows_seen += other.rows_seen
        self.filtered_rows += other.filtered_rows
        self.exams_kept += other.exams_kept
        self.drops.update(other.drops)
        self.unmapped.update(other.unmapped)

    def absorb_unmapped(self, counts) -> None:
        merge_unmapped_counts(self.unmapped, counts)

    def render(self) -> str:
        dropped = sum(self.drops.values())
        lines = [
            f"=== BUILD REPORT: {self.dataset} ===",
            f"  source rows seen : {self.rows_seen:,}",
            f"  source rows filtered : {self.filtered_rows:,}",
            f"  exams written    : {self.exams_kept:,}",
            f"  processing units dropped : {dropped:,}",
        ]
        for reason, count in self.drops.most_common():
            lines.append(f"      {count:>8,}  {reason}")
        lines.append("  " + format_unmapped_report(self.unmapped).replace("\n", "\n  "))
        return "\n".join(lines)


class BreastCancerDataset(ABC):
    def __init__(self, csv_save_path: str):
        self.save_path = csv_save_path
        self.data = []
        self.dataset_name = None
        self.report = BuildReport()

    @abstractmethod
    def process_info(self):
        raise NotImplementedError("Subclasses must implement process_info()")

    def get_dataset_name(self) -> str:
        assert self.dataset_name is not None, "Dataset name is not set"
        return self.dataset_name

    def set_dataset_name(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.report.dataset = dataset_name

    def append_exam(self, exam: ExamInformation | None, reason: str = "unknown error"):
        if exam is None:
            self.report.record_drop(reason)
            return
        self.data.append(exam.to_row())
        self.report.exams_kept += 1

    def append_exams(self, exams) -> None:
        for exam in exams or []:
            self.append_exam(exam)

    def collect_pool_results(self, results) -> None:
        """Absorb exams, unmapped counts, and drop counts from each worker."""
        for exams, counts, drops in results:
            self.report.absorb_unmapped(counts)
            self.report.drops.update(drops)
            if exams is None:
                continue
            self.append_exams(exams)

    # -- multiprocessing -----------------------------------------------------
    def run_pool(self, worker, items, desc: str) -> None:
        """Map `worker` over `items` in a process pool and collect the results.

        `worker` must be decorated with `pool_worker`.
        """
        rows = sum(len(item) if isinstance(item, pd.DataFrame) else 1 for item in items)
        self.report.rows_seen = self.report.rows_seen or rows
        self.report.filtered_rows += max(0, self.report.rows_seen - rows)
        with Pool(processes=_worker_count()) as pool:
            results = list(
                tqdm(
                    pool.imap(worker, items),
                    total=len(items),
                    desc=desc,
                    unit="item",
                    ncols=100,
                    leave=True,
                )
            )
        self.collect_pool_results(results)

    def run_row_pool(self, df: pd.DataFrame) -> None:
        """Split `df` one batch per worker and run `self.process_row` on every row.

        The subclass supplies `process_row(row)`, returning an `ExamInformation`,
        a list of them, or None.
        """
        self.report.rows_seen = self.report.rows_seen or len(df)
        self.report.filtered_rows += max(0, self.report.rows_seen - len(df))
        n = _worker_count()
        batches = [
            df.iloc[idx]
            for idx in np.array_split(np.arange(len(df)), n)
            if len(idx) > 0
        ]
        with Pool(processes=n) as pool:
            results = pool.map(self._process_batch, batches)
        self.collect_pool_results(results)

    def process_row(self, row) -> ExamInformation | list[ExamInformation] | None:
        """Turn one dataframe row into an exam. Required only by `run_row_pool`.

        Not abstract: an adapter whose unit of work is not a row -- a patient, an
        image with several findings, a record read straight out of a .mat file --
        drives its own worker through `run_pool` and never implements this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} reached run_row_pool without implementing "
            "process_row(); implement it, or dispatch with run_pool instead."
        )

    @pool_worker
    def _process_batch(self, df: pd.DataFrame):
        exams = []
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Processing {self.get_dataset_name()} batch",
            unit="exam",
            ncols=100,
            leave=True,
        ):
            try:
                result = self.process_row(row)
            except Exception as exc:
                logger.error(f"Error processing row {row.name}: {exc}")
                self.report.record_drop("row processing failed")
                continue
            if not result:
                self.report.record_drop("no usable exam")
                continue
            exams.extend(result if isinstance(result, list) else [result])
        return exams

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    def save_csv(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        df = self.to_dataframe()
        df.to_csv(self.save_path, index=False)

        parquet_path = os.path.splitext(self.save_path)[0] + ".parquet"
        try:
            df.to_parquet(parquet_path, index=False)
        except Exception as exc:
            logger.warning(f"Could not write {parquet_path}: {exc}")
            if os.path.exists(parquet_path):
                os.remove(parquet_path)

        self.report.absorb_unmapped(drain_unmapped_counts())
        logger.info("\n%s", self.report.render())
        report_path = os.path.splitext(self.save_path)[0] + "_build_report.txt"
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(self.report.render() + "\n")
