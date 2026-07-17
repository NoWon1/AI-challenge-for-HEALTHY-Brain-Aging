"""Minimal schema validation for tabular cohort files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


DEFAULT_CONTRACT: dict[str, Any] = {
    "tables": {
        "participants": {
            "primary_key": "participant_id",
            "required_columns": {"participant_id": "string", "cohort": "string"},
            "recommended_columns": {
                "sex": "string",
                "birth_year": "integer",
                "education_years": "number",
                "language": "string",
                "urban_rural": "string",
            },
        },
        "visits": {
            "primary_key": "visit_id",
            "required_columns": {
                "visit_id": "string",
                "participant_id": "string",
                "cohort": "string",
                "visit_index": "integer",
                "age_at_visit": "number",
            },
            "recommended_columns": {
                "visit_date": "date",
                "baseline_days": "integer",
                "diagnosis": "string",
                "cdr_global": "number",
                "cognitive_status": "string",
            },
        },
        "modality_features": {
            "primary_key": "feature_row_id",
            "required_columns": {
                "feature_row_id": "string",
                "participant_id": "string",
                "visit_id": "string",
                "cohort": "string",
                "modality": "string",
                "feature_name": "string",
                "value": "number",
            },
            "recommended_columns": {
                "unit": "string",
                "source_variable": "string",
                "qc_flag": "string",
                "derived": "boolean",
            },
        },
        "outcomes": {
            "primary_key": "outcome_id",
            "required_columns": {
                "outcome_id": "string",
                "participant_id": "string",
                "anchor_visit_id": "string",
                "endpoint": "string",
                "horizon_days": "integer",
                "event": "integer",
            },
            "recommended_columns": {
                "event_time_days": "integer",
                "future_score": "number",
                "censoring_reason": "string",
            },
        },
    }
}


@dataclass(frozen=True)
class TableContract:
    name: str
    primary_key: str
    required_columns: dict[str, str]
    recommended_columns: dict[str, str]

    def missing_required(self, columns: set[str]) -> list[str]:
        return [column for column in self.required_columns if column not in columns]


def load_contract(path: str | Path | None = None) -> dict[str, TableContract]:
    contract_path = Path(path) if path else Path(__file__).with_name("common_data_model.yaml")
    if yaml is None:
        if path is not None:
            raise RuntimeError("pyyaml is required to load custom YAML data contracts")
        payload = DEFAULT_CONTRACT
    else:
        payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    tables = payload.get("tables", {})
    return {
        name: TableContract(
            name=name,
            primary_key=definition["primary_key"],
            required_columns=definition.get("required_columns", {}),
            recommended_columns=definition.get("recommended_columns", {}),
        )
        for name, definition in tables.items()
    }


def validate_columns(table_name: str, columns: list[str], contract_path: str | Path | None = None) -> None:
    contracts = load_contract(contract_path)
    if table_name not in contracts:
        raise KeyError(f"Unknown table contract: {table_name}")
    missing = contracts[table_name].missing_required(set(columns))
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {', '.join(missing)}")
