import json
from dataclasses import asdict, dataclass
from typing import Optional

from utils.preprocessing import NOT_PRESENT, sanitize_string


# ACR BI-RADS COMPLIANT
@dataclass
class BreastFindings:
    density: Optional[str] = None  # mg, mr
    tissue_composition: Optional[str] = None  # us


@dataclass
class CalcificationAbnormality:
    type: Optional[str] = None
    type_details: Optional[str] = None
    distribution: Optional[str] = None


@dataclass
class MassAbnormality:
    shape: Optional[str] = None  # all
    margin: Optional[str] = None  # all

    # mg, mr
    density: Optional[str] = None

    # us
    echogenicity: Optional[str] = None
    posterior_features: Optional[str] = None
    halo: Optional[str] = None


@dataclass
class Lesion:
    laterality: Optional[str] = (
        None  # "left" | "right" => currently useful just for MRIs since MGs and USs get only an image of one of of the breasts. Only in bilateral MRIs we have both breasts imaged.
    )
    location: Optional[str] = None  # "upper outer quadrant", etc.
    type: Optional[str] = None  # "mass" | "calcification" | "non mass enhancement"
    mass_details: Optional[MassAbnormality] = None
    calcification_details: Optional[CalcificationAbnormality] = None

    def __post_init__(self):
        if self.type is not None and self.type == "mass":
            assert (
                self.calcification_details is None
            ), "A mass lesion should not have calcification details defined."
        elif self.type is not None and self.type == "calcification":
            assert (
                self.mass_details is None
            ), "A calcification lesion should not have mass details."


@dataclass
class Assessment:
    birads: Optional[str] = None


class JSONReportFindings:
    def __init__(
        self,
        breast: Optional[BreastFindings] = None,
        lesion: Optional[
            Lesion | str
        ] = None,  # NOTE: if the value for lesion is None => we dont know the value so we do not ask the model to predict it. if value is "NOT_PRESENT" => we know the value and it is negative, so when asked, the model should say there are no lesions.
        assessment: Optional[Assessment] = None,
    ) -> None:
        self.breast_findings = breast
        self.lesion = lesion
        self.assessment = assessment
        self.context = None

    def _clean(self, data: dict) -> dict:
        res = {}
        for k, v in data.items():
            if isinstance(v, dict):
                v = self._clean(v)
            elif isinstance(v, list):
                v = [self._clean(item) for item in v]

            if v is None or len(v) == 0:
                continue

            res[sanitize_string(k)] = v
        return res

    def construct_context(self) -> None:
        context_dict = {}

        # add breast findings if given
        if self.breast_findings is not None:
            context_dict["breast"] = self._clean(asdict(self.breast_findings))

        # add lesion if given
        if self.lesion is not None:
            if self.lesion == NOT_PRESENT:
                context_dict["lesion"] = NOT_PRESENT
            else:
                assert isinstance(
                    self.lesion, Lesion
                ), "If lesion is not None, it must be an instance of Lesion or the string NOT_PRESENT."
                context_dict["lesion"] = self._clean(asdict(self.lesion))

        # add assessment if given
        if self.assessment is not None:
            assert (
                self.assessment.birads is not None
            ), "If assessment is given, birads must be defined."
            context_dict["assessment"] = self._clean(asdict(self.assessment))

        self.context = context_dict

    def get_string(self) -> str:
        if self.context is None:
            self.construct_context()
        return json.dumps(self.context, separators=(",", ":"))
