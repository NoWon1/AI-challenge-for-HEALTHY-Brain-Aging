"""Cohort-neutral longitudinal records with scientific invariants."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from neurosaarthi.core.errors import DataValidationError


class EquivalenceLevel(StrEnum):
    EXACT = "EXACT"
    COMPATIBLE = "COMPATIBLE"
    DERIVED = "DERIVED"
    APPROXIMATE = "APPROXIMATE"
    NON_EQUIVALENT = "NON_EQUIVALENT"


def _require_text(name: str, value: str) -> None:
    if not value or not value.strip():
        raise DataValidationError(f"{name} must be a non-empty string")


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise DataValidationError(f"{name} must be finite")


@dataclass(frozen=True)
class Participant:
    participant_id_internal: str
    cohort: str
    site: str | None = None
    sex: str | None = None
    birth_year_or_age: float | None = None
    education_years: float | None = None
    ancestry_group_if_permitted: str | None = None

    def __post_init__(self) -> None:
        _require_text("participant_id_internal", self.participant_id_internal)
        _require_text("cohort", self.cohort)
        if self.education_years is not None and not 0 <= self.education_years <= 50:
            raise DataValidationError("education_years must be between 0 and 50")


@dataclass(frozen=True)
class Visit:
    participant_id_internal: str
    visit_id: str
    visit_index: int
    time_from_baseline_days: int
    age_at_visit: float
    diagnostic_state: str | None = None

    def __post_init__(self) -> None:
        _require_text("participant_id_internal", self.participant_id_internal)
        _require_text("visit_id", self.visit_id)
        if self.visit_index < 0:
            raise DataValidationError("visit_index must be non-negative")
        if self.time_from_baseline_days < 0:
            raise DataValidationError("time_from_baseline_days must be non-negative")
        _require_finite("age_at_visit", self.age_at_visit)
        if not 0 < self.age_at_visit < 130:
            raise DataValidationError("age_at_visit must be in (0, 130)")


@dataclass(frozen=True)
class ImagingRecord:
    participant_id_internal: str
    visit_id: str
    modality: str
    sequence: str
    image_uri: str
    scanner_vendor: str | None = None
    field_strength_t: float | None = None
    voxel_spacing_mm: tuple[float, float, float] | None = None
    orientation: str | None = None
    qc_status: str = "pending"

    def __post_init__(self) -> None:
        for name in ("participant_id_internal", "visit_id", "modality", "sequence", "image_uri"):
            _require_text(name, getattr(self, name))
        if self.image_uri.lower().startswith(("http://", "https://", "s3://", "gs://")):
            raise DataValidationError("Remote image URIs are disabled; use an approved local secure path")
        if self.field_strength_t is not None and self.field_strength_t <= 0:
            raise DataValidationError("field_strength_t must be positive")
        if self.voxel_spacing_mm is not None and any(item <= 0 for item in self.voxel_spacing_mm):
            raise DataValidationError("voxel_spacing_mm values must be positive")


@dataclass(frozen=True)
class CognitiveObservation:
    participant_id_internal: str
    visit_id: str
    instrument: str
    domain: str
    raw_score: float | None
    normalized_score: float | None = None
    language: str | None = None
    education_adjustment: str | None = None

    def __post_init__(self) -> None:
        for name in ("participant_id_internal", "visit_id", "instrument", "domain"):
            _require_text(name, getattr(self, name))
        for name in ("raw_score", "normalized_score"):
            value = getattr(self, name)
            if value is not None:
                _require_finite(name, value)


@dataclass(frozen=True)
class FeatureProvenance:
    feature_name: str
    source_dataset: str
    original_variable: str
    unit: str
    canonical_unit: str
    transform: str
    availability: str
    missingness_code: str
    version: str
    equivalence: EquivalenceLevel
    evidence: str

    def __post_init__(self) -> None:
        for name in (
            "feature_name",
            "source_dataset",
            "original_variable",
            "unit",
            "canonical_unit",
            "transform",
            "availability",
            "missingness_code",
            "version",
            "evidence",
        ):
            _require_text(name, getattr(self, name))
        if (
            self.equivalence is EquivalenceLevel.NON_EQUIVALENT
            and self.feature_name == self.original_variable
        ):
            raise DataValidationError(
                "NON_EQUIVALENT variables cannot silently retain the source variable name"
            )
