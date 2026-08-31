import json
from dataclasses import asdict, dataclass

from utils.preprocessing import sanitize_string


@dataclass
class PatientContext:
    age: str | None = None
    has_implants: str | None = None


@dataclass
class ExamContext:
    modality: str | None = None  # mg, us, mr
    laterality: str | None = None  # left, right, bilateral
    view: str | None = None  # cranial caudal, mediolateral oblique, etc.
    contrast_phase: str | None = None  # pre-contrast, post-contrast, etc.


class JSONReportContext:
    def __init__(
            self,
            patient_context: PatientContext | None = None,
            exam_context: ExamContext | None = None,
    ) -> None:
        self.patient_context = patient_context
        self.exam_context = exam_context
        self.context = None

    def _clean(self, d: dict) -> dict:
        return {sanitize_string(k): v for k, v in d.items() if v is not None}

    def construct_context(self) -> None:
        acc = {}

        if self.patient_context is not None:
            patient_dict = self._clean(asdict(self.patient_context))
            if patient_dict:
                acc["patient"] = patient_dict

        if self.exam_context is not None:
            exam_dict = self._clean(asdict(self.exam_context))
            if exam_dict:
                acc["exam"] = exam_dict

        self.context = acc

    def get_string(self) -> str:
        if self.context is None:
            self.construct_context()
        return json.dumps(self.context, separators=(",", ":"))
