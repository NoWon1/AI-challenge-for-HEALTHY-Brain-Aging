"""Configuration-driven OASIS-3 session-manifest adapter.

OASIS-3 releases and authorised exports can differ. The adapter therefore
requires an explicit column mapping and never guesses a diagnosis, date, or
cognitive-test equivalence.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import yaml

from neurosaarthi.core.errors import ConfigurationError, DataValidationError


@dataclass(frozen=True)
class Oasis3Columns:
    participant_id: str
    visit_id: str
    age_at_visit: str
    time_from_baseline_days: str
    image_uri: str
    sequence: str
    site: str | None = None
    sex: str | None = None
    education_years: str | None = None
    diagnosis: str | None = None
    cognitive_instrument: str | None = None
    cognitive_score: str | None = None
    scanner_vendor: str | None = None
    field_strength_t: str | None = None


@dataclass(frozen=True)
class Oasis3Tables:
    participants: pd.DataFrame
    visits: pd.DataFrame
    imaging: pd.DataFrame
    cognitive: pd.DataFrame
    provenance: pd.DataFrame


def pseudonymize_identifier(raw_identifier: str, secret: str, prefix: str = "OAS3") -> str:
    """Create a stable keyed pseudonym without persisting an identifier map."""

    if len(secret) < 16:
        raise ConfigurationError("Pseudonymisation secret must contain at least 16 characters")
    # BLAKE2 keys are limited to 64 bytes. Deriving a fixed-size binary key also
    # avoids treating a long Unicode secret's character count as its byte count.
    derived_key = hashlib.sha256(secret.encode("utf-8")).digest()
    digest = hashlib.blake2b(raw_identifier.encode("utf-8"), key=derived_key, digest_size=12).hexdigest()
    return f"{prefix}-{digest}"


class Oasis3ManifestAdapter:
    """Map an authorised local OASIS-3 manifest to the canonical tables."""

    cohort_name = "OASIS-3"

    def __init__(self, manifest_path: str | Path, columns: Oasis3Columns, pseudonymisation_secret: str):
        self.manifest_path = Path(manifest_path).resolve()
        self.columns = columns
        self.pseudonymisation_secret = pseudonymisation_secret

    @classmethod
    def from_yaml(cls, path: str | Path, pseudonymisation_secret: str) -> Oasis3ManifestAdapter:
        config_path = Path(path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict) or not isinstance(payload.get("columns"), dict):
            raise ConfigurationError("OASIS-3 adapter config requires a columns mapping")
        try:
            columns = Oasis3Columns(**payload["columns"])
            manifest_path = Path(payload["manifest_path"])
        except (KeyError, TypeError) as exc:
            raise ConfigurationError("Invalid OASIS-3 adapter configuration") from exc
        if not manifest_path.is_absolute():
            manifest_path = config_path.parent / manifest_path
        return cls(manifest_path, columns, pseudonymisation_secret)

    def extract(self) -> Oasis3Tables:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"OASIS-3 manifest not found: {self.manifest_path}")
        raw = pd.read_csv(self.manifest_path, low_memory=False)
        self._validate_columns(raw)
        frame = raw.copy()
        source_id = frame[self.columns.participant_id].astype("string")
        if source_id.isna().any() or source_id.str.strip().eq("").any():
            raise DataValidationError("OASIS-3 participant identifiers must not be missing")
        frame["participant_id_internal"] = source_id.map(
            lambda item: pseudonymize_identifier(str(item), self.pseudonymisation_secret)
        )
        # Ordering must not depend on the secret-derived pseudonym. Retain the
        # manifest's participant order while sorting visits chronologically.
        participant_order = {
            participant_id: index
            for index, participant_id in enumerate(frame["participant_id_internal"].drop_duplicates())
        }
        frame["_participant_order"] = frame["participant_id_internal"].map(participant_order)
        frame["visit_id_internal"] = [
            pseudonymize_identifier(f"{subject}|{visit}", self.pseudonymisation_secret, prefix="VIS")
            for subject, visit in zip(source_id, frame[self.columns.visit_id], strict=True)
        ]
        frame["age_at_visit"] = pd.to_numeric(frame[self.columns.age_at_visit], errors="coerce")
        frame["time_from_baseline_days"] = pd.to_numeric(
            frame[self.columns.time_from_baseline_days], errors="coerce"
        )
        if frame[["age_at_visit", "time_from_baseline_days"]].isna().any().any():
            raise DataValidationError("Age and time-from-baseline must be numeric and complete")
        if (frame["time_from_baseline_days"] < 0).any():
            raise DataValidationError("time_from_baseline_days cannot be negative")
        frame = frame.sort_values(["_participant_order", "time_from_baseline_days"], kind="stable")
        frame["visit_index"] = frame.groupby("participant_id_internal").cumcount()
        self._validate_visit_order(frame)

        participants = self._participants(frame)
        visits = self._visits(frame)
        imaging = self._imaging(frame)
        cognitive = self._cognitive(frame)
        provenance = self._provenance()
        return Oasis3Tables(participants, visits, imaging, cognitive, provenance)

    def _validate_columns(self, frame: pd.DataFrame) -> None:
        mapping = asdict(self.columns)
        required = {
            value
            for key, value in mapping.items()
            if value is not None and key not in {"cognitive_instrument", "cognitive_score"}
        }
        cognitive_pair = (self.columns.cognitive_instrument, self.columns.cognitive_score)
        if (cognitive_pair[0] is None) != (cognitive_pair[1] is None):
            raise ConfigurationError("cognitive_instrument and cognitive_score must be mapped together")
        required.update(value for value in cognitive_pair if value)
        missing = sorted(required - set(frame.columns))
        if missing:
            raise DataValidationError(f"OASIS-3 manifest is missing mapped columns: {', '.join(missing)}")

    @staticmethod
    def _validate_visit_order(frame: pd.DataFrame) -> None:
        duplicates = frame.duplicated(["participant_id_internal", "visit_id_internal"])
        if duplicates.any():
            raise DataValidationError("Duplicate participant/visit rows are not allowed")
        monotonic = frame.groupby("participant_id_internal")["time_from_baseline_days"].apply(
            lambda values: values.is_monotonic_increasing
        )
        if not bool(monotonic.all()):
            raise DataValidationError("Visit time must be monotonic within participant")

    def _participants(self, frame: pd.DataFrame) -> pd.DataFrame:
        baseline = frame.drop_duplicates("participant_id_internal", keep="first")
        result = pd.DataFrame(
            {
                "participant_id_internal": baseline["participant_id_internal"],
                "cohort": self.cohort_name,
                "site": _optional_series(baseline, self.columns.site),
                "sex": _optional_series(baseline, self.columns.sex),
                "education_years": _optional_numeric(baseline, self.columns.education_years),
            }
        )
        return result.reset_index(drop=True)

    def _visits(self, frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "participant_id_internal": frame["participant_id_internal"],
                "visit_id": frame["visit_id_internal"],
                "visit_index": frame["visit_index"].astype(int),
                "time_from_baseline_days": frame["time_from_baseline_days"].astype(int),
                "age_at_visit": frame["age_at_visit"].astype(float),
                "diagnostic_state": _optional_series(frame, self.columns.diagnosis),
            }
        ).reset_index(drop=True)

    def _imaging(self, frame: pd.DataFrame) -> pd.DataFrame:
        uris = frame[self.columns.image_uri].astype("string")
        if uris.str.lower().str.startswith(("http://", "https://", "s3://", "gs://")).any():
            raise DataValidationError("Remote image URIs are prohibited")
        return pd.DataFrame(
            {
                "participant_id_internal": frame["participant_id_internal"],
                "visit_id": frame["visit_id_internal"],
                "modality": "MRI",
                "sequence": frame[self.columns.sequence].astype("string"),
                "image_uri": uris,
                "scanner_vendor": _optional_series(frame, self.columns.scanner_vendor),
                "field_strength_t": _optional_numeric(frame, self.columns.field_strength_t),
                "qc_status": "pending",
            }
        ).reset_index(drop=True)

    def _cognitive(self, frame: pd.DataFrame) -> pd.DataFrame:
        if self.columns.cognitive_instrument is None or self.columns.cognitive_score is None:
            return pd.DataFrame(
                columns=[
                    "participant_id_internal",
                    "visit_id",
                    "instrument",
                    "domain",
                    "raw_score",
                    "normalized_score",
                ]
            )
        return pd.DataFrame(
            {
                "participant_id_internal": frame["participant_id_internal"],
                "visit_id": frame["visit_id_internal"],
                "instrument": frame[self.columns.cognitive_instrument].astype("string"),
                "domain": "instrument_total",
                "raw_score": pd.to_numeric(frame[self.columns.cognitive_score], errors="coerce"),
                "normalized_score": pd.NA,
            }
        ).reset_index(drop=True)

    def _provenance(self) -> pd.DataFrame:
        rows = [
            {
                "feature_name": "age_at_visit",
                "source_dataset": self.cohort_name,
                "original_variable": self.columns.age_at_visit,
                "unit": "years",
                "canonical_unit": "years",
                "transform": "numeric_cast",
                "availability": "manifest",
                "missingness_code": "NA",
                "version": "oasis3-adapter-v1",
                "equivalence": "EXACT",
                "evidence": "Explicit user-provided column mapping",
            },
            {
                "feature_name": "time_from_baseline_days",
                "source_dataset": self.cohort_name,
                "original_variable": self.columns.time_from_baseline_days,
                "unit": "days",
                "canonical_unit": "days",
                "transform": "numeric_cast",
                "availability": "manifest",
                "missingness_code": "NA",
                "version": "oasis3-adapter-v1",
                "equivalence": "EXACT",
                "evidence": "Explicit user-provided column mapping",
            },
        ]
        return pd.DataFrame(rows)


def _optional_series(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    return frame[column].astype("string")


def _optional_numeric(frame: pd.DataFrame, column: str | None) -> pd.Series:
    if column is None:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")
