from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from utils.preprocessing import (
    ALL_FINDING_TYPES,
    CALCIFICATION,
    FINDING_TYPE_DETAILS,
    MASS,
    NOT_PRESENT,
    PRESENT,
    sanitize_string,
)

LESION_STATUS_UNKNOWN = "unknown"
LESION_STATUS_ABSENT = "absent"
LESION_STATUS_PRESENT = "present"
LESION_STATUSES = (
    LESION_STATUS_UNKNOWN,
    LESION_STATUS_ABSENT,
    LESION_STATUS_PRESENT,
)

SCOPE_COMPLETE = "complete"
SCOPE_INDEX_ONLY = "index_only"
SCOPE_UNKNOWN = "unknown"
ANNOTATION_SCOPES = (SCOPE_COMPLETE, SCOPE_INDEX_ONLY, SCOPE_UNKNOWN)


@dataclass
class BreastFindings:
    # Mammography: breast composition (a)-(d)
    composition: str | None = None

    # MRI: amount of fibroglandular tissue (a)-(d)
    fgt: str | None = None

    # Ultrasound: background echotexture
    tissue_composition: str | None = None


# ===========================================================================
# FINDING DESCRIPTOR PAYLOADS
# ===========================================================================
@dataclass
class MassAbnormality:
    # all modalities
    shape: str | None = None
    margin: str | None = None

    # mammography, MRI
    density: str | None = None

    # ultrasound
    echo_pattern: str | None = None
    posterior_features: str | None = None
    echogenic_feature: str | None = None


@dataclass
class CalcificationAbnormality:
    type: str | None = None
    type_details: str | None = None
    distribution: str | None = None


# ===========================================================================
# LOCATION
# ===========================================================================
@dataclass
class LesionLocation:
    laterality: str | None = None
    clock_position: int | None = None
    quadrant: str | None = None  # "upper outer", etc.
    special_location: str | None = None  # "retroareolar" | "central" | "axillary tail"
    coarse_location: str | None = None  # "upper" | "medial", etc. (not a v2025 term)
    depth: str | None = None  # "anterior third", etc.

    def is_empty(self) -> bool:
        return all(getattr(self, f) is None for f in self.__dataclass_fields__)

    def describe(self) -> str | None:
        parts: list[str] = []
        if self.clock_position is not None:
            parts.append(f"{self.clock_position} o'clock")
        elif self.quadrant is not None:
            parts.append(f"the {self.quadrant} quadrant")
        elif self.special_location is not None:
            parts.append(f"the {self.special_location} region")
        elif self.coarse_location is not None:
            parts.append(f"the {self.coarse_location} breast")

        if self.depth is not None:
            parts.append(self.depth)

        if self.laterality in ("left", "right"):
            parts.append(f"the {self.laterality} breast")

        return ", ".join(parts) if parts else None


# ===========================================================================
# A SINGLE FINDING
# ===========================================================================
_DETAIL_FIELDS = (
    "mass_details",
    "calcification_details",
)

_declared = set(FINDING_TYPE_DETAILS.values())
if not _declared <= set(_DETAIL_FIELDS):
    raise AssertionError(
        f"FINDING_TYPE_DETAILS names payload(s) {sorted(_declared - set(_DETAIL_FIELDS))} "
        f"that Lesion does not define. Lesion has {list(_DETAIL_FIELDS)}."
    )


@dataclass
class Lesion:
    finding_type: str | None = None
    location: LesionLocation | None = None

    mass_details: MassAbnormality | None = None
    calcification_details: CalcificationAbnormality | None = None
    is_index: bool = False  # for index lesion (most important finding)
    lesion_id: str | None = None

    def __post_init__(self):
        if self.finding_type is not None and self.finding_type not in ALL_FINDING_TYPES:
            raise ValueError(
                f"Unknown finding type {self.finding_type!r}. Expected one of "
                f"{list(ALL_FINDING_TYPES)}. Route legacy labels through "
                "get_proper_finding_type() first."
            )

        set_details = [f for f in _DETAIL_FIELDS if getattr(self, f) is not None]
        if len(set_details) > 1:
            raise ValueError(
                f"A finding carries at most one descriptor payload; got {set_details}."
            )
        if not set_details:
            return

        expected = FINDING_TYPE_DETAILS.get(self.finding_type)
        if expected is None:
            raise ValueError(
                f"Finding type {self.finding_type!r} takes no descriptor payload, "
                f"but {set_details[0]} was set."
            )
        if set_details[0] != expected:
            raise ValueError(
                f"Finding type {self.finding_type!r} expects {expected}, "
                f"got {set_details[0]}."
            )

    def details(self):
        for name in _DETAIL_FIELDS:
            value = getattr(self, name)
            if value is not None:
                return name, value
        return None, None

    def has_descriptors(self) -> bool:
        _, payload = self.details()
        if payload is None:
            return False
        return any(v is not None and v is not False for v in asdict(payload).values())

    def referent(self, ordinal: int | None = None) -> str:
        noun = self.finding_type or "finding"
        if noun == CALCIFICATION:
            noun = "group of calcifications"

        qualifiers: list[str] = []
        if self.finding_type == MASS and self.mass_details is not None:
            for attribute in (self.mass_details.margin, self.mass_details.shape):
                if attribute is not None:
                    qualifiers.append(attribute)
                    break

        where = self.location.describe() if self.location is not None else None

        phrase = f"the {' '.join(qualifiers + [noun])}"
        if where:
            phrase = f"{phrase} at {where}"
        elif ordinal is not None:
            phrase = f"{phrase} (finding {ordinal})"
        return phrase


@dataclass
class Assessment:
    birads: str | None = None


# ===========================================================================
# THE RECORD
# ===========================================================================
@dataclass
class JSONReportFindings:
    breast: BreastFindings | None = None
    lesions: list[Lesion] = field(default_factory=list)
    lesion_status: str = LESION_STATUS_UNKNOWN
    annotation_scope: str = SCOPE_UNKNOWN
    assessment: Assessment | None = None

    def __init__(self):
        self.context = None

    def __post_init__(self):
        if self.lesion_status not in LESION_STATUSES:
            raise ValueError(
                f"lesion_status must be one of {list(LESION_STATUSES)}, "
                f"got {self.lesion_status!r}"
            )
        if self.annotation_scope not in ANNOTATION_SCOPES:
            raise ValueError(
                f"annotation_scope must be one of {list(ANNOTATION_SCOPES)}, "
                f"got {self.annotation_scope!r}"
            )
        if self.lesions and self.lesion_status != LESION_STATUS_PRESENT:
            raise ValueError(
                f"{len(self.lesions)} findings were given but lesion_status is "
                f"{self.lesion_status!r}; a described finding is a present one."
            )
        if (
                self.lesion_status == LESION_STATUS_ABSENT
                and self.annotation_scope
                not in (
                SCOPE_COMPLETE,
                SCOPE_UNKNOWN,
        )
        ):
            raise ValueError(
                "An image with no findings cannot have an index-only annotation scope."
            )

        for position, lesion in enumerate(self.lesions, start=1):
            if lesion.lesion_id is None:
                lesion.lesion_id = f"L{position}"

        if self.lesions and not any(lesion.is_index for lesion in self.lesions):
            self.lesions[0].is_index = True

        self.context: dict | None = None

    # -- construction helpers ------------------------------------------------
    @staticmethod
    def resolve_lesions(value) -> tuple[list[Lesion], str]:
        """Normalise an adapter's finding argument into (lesions, lesion_status).

        Accepts what the sources naturally produce:
          None                  -> we do not know whether anything is there
          "not_present"         -> we know there is nothing
          "present"             -> we know something is there, no descriptors
          Lesion                -> one described finding
          list[Lesion]          -> several described findings (may be empty,
                                   which means the same as "not_present")
        """
        if value is None:
            return [], LESION_STATUS_UNKNOWN
        if isinstance(value, str):
            if value == NOT_PRESENT:
                return [], LESION_STATUS_ABSENT
            if value == PRESENT:
                return [], LESION_STATUS_PRESENT
            raise ValueError(
                f"Unknown lesion sentinel {value!r}; expected NOT_PRESENT or PRESENT."
            )
        if isinstance(value, Lesion):
            return [value], LESION_STATUS_PRESENT
        if isinstance(value, (list, tuple)):
            lesions = [item for item in value if item is not None]
            if not all(isinstance(item, Lesion) for item in lesions):
                raise TypeError("Every element of a lesion list must be a Lesion.")
            if not lesions:
                return [], LESION_STATUS_ABSENT
            return list(lesions), LESION_STATUS_PRESENT
        raise TypeError(f"Cannot interpret {type(value).__name__} as findings.")

    @classmethod
    def build(
            cls,
            breast: BreastFindings | None = None,
            lesion=None,
            assessment: Assessment | None = None,
            annotation_scope: str = SCOPE_UNKNOWN,
    ) -> JSONReportFindings:
        lesions, status = cls.resolve_lesions(lesion)
        if status != LESION_STATUS_PRESENT:
            annotation_scope = (
                SCOPE_COMPLETE if status == LESION_STATUS_ABSENT else SCOPE_UNKNOWN
            )
        return cls(
            breast=breast,
            lesions=lesions,
            lesion_status=status,
            annotation_scope=annotation_scope,
            assessment=assessment,
        )

    # -- serialisation -------------------------------------------------------
    def _clean(self, data):
        if isinstance(data, dict):
            out = {}
            for key, value in data.items():
                cleaned = self._clean(value)
                if cleaned is None:
                    continue
                if isinstance(cleaned, (str, dict, list)) and len(cleaned) == 0:
                    continue
                out[sanitize_string(key)] = cleaned
            return out
        if isinstance(data, list):
            return [self._clean(item) for item in data]
        return data

    def _lesion_dict(self, lesion: Lesion) -> dict:
        payload = self._clean(asdict(lesion))
        payload["is index"] = lesion.is_index
        payload["lesion id"] = lesion.lesion_id
        return payload

    def construct_context(self) -> None:
        out: dict = {}

        if self.breast is not None:
            cleaned = self._clean(asdict(self.breast))
            if cleaned:
                out["breast"] = cleaned

        out["lesion status"] = self.lesion_status
        if self.lesion_status == LESION_STATUS_PRESENT:
            out["annotation scope"] = self.annotation_scope
            if self.lesions:
                out["lesions"] = [self._lesion_dict(item) for item in self.lesions]

        if self.assessment is not None:
            if self.assessment.birads is None:
                raise ValueError("If an assessment is given, birads must be defined.")
            cleaned = self._clean(asdict(self.assessment))
            if cleaned:
                out["assessment"] = cleaned

        self.context = out

    def to_dict(self) -> dict:
        if self.context is None:
            self.construct_context()
        return self.context or {}

    def get_string(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"))
