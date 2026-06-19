import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import pandas as pd


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
    segmentation: Optional[list[str]]  # path for the segmentations (if any)

    context: str  # context that might be available/relevant for findings, e.g. demographics, genetic info, etc...
    findings: (
        str  # findings for the exam, essentially, information to infer from the images
    )


class BreastCancerDataset(ABC):
    def __init__(self, csv_save_path: str):
        self.save_path = csv_save_path
        self.data = []
        self.dataset_name = None

    @abstractmethod
    def process_info(self):
        pass

    def get_dataset_name(self) -> str:
        assert self.dataset_name is not None, "Dataset name is not set"
        return self.dataset_name

    def set_dataset_name(self, dataset_name: str):
        self.dataset_name = dataset_name

    def append_exam(self, exam: Optional[ExamInformation]):
        if exam is not None:
            self.data.append(exam.__dict__)

    def save_csv(self):
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        df = pd.DataFrame(self.data)
        df.to_csv(self.save_path, index=False)
