"""Typed settings with privacy-preserving defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

from neurosaarthi.core.errors import ConfigurationError, DataGovernanceError


class Environment(StrEnum):
    """Execution environments ordered by data sensitivity, not capability."""

    LOCAL_PUBLIC = "local_public"
    CI_SYNTHETIC = "ci_synthetic"
    SECURE_CBR = "secure_cbr"


PROTECTED_COHORTS = frozenset({"CBR-SANSCOG", "CBR-TLSA", "SANSCOG", "TLSA"})


@dataclass(frozen=True)
class Settings:
    """Runtime settings.

    Network access is disabled by default in every environment. Enabling it in
    ``secure_cbr`` is rejected because protected participant-level data must
    remain within the approved secure environment.
    """

    environment: Environment = Environment.LOCAL_PUBLIC
    data_root: Path = Path("data")
    allow_network: bool = False
    log_level: str = "INFO"
    random_seed: int = 42

    def __post_init__(self) -> None:
        if self.environment is Environment.SECURE_CBR and self.allow_network:
            raise DataGovernanceError("Network access cannot be enabled in the secure CBR environment")
        if self.random_seed < 0:
            raise ConfigurationError("random_seed must be non-negative")
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ConfigurationError(f"Unsupported log level: {self.log_level}")

    @classmethod
    def from_environment(cls) -> Settings:
        """Load settings from ``NEUROSAARTHI_*`` environment variables."""

        environment_text = os.getenv("NEUROSAARTHI_ENVIRONMENT", Environment.LOCAL_PUBLIC.value)
        try:
            environment = Environment(environment_text)
        except ValueError as exc:
            raise ConfigurationError(f"Unknown environment: {environment_text}") from exc
        return cls(
            environment=environment,
            data_root=Path(os.getenv("NEUROSAARTHI_DATA_ROOT", "data")),
            allow_network=_parse_bool(os.getenv("NEUROSAARTHI_ALLOW_NETWORK", "false")),
            log_level=os.getenv("NEUROSAARTHI_LOG_LEVEL", "INFO").upper(),
            random_seed=int(os.getenv("NEUROSAARTHI_RANDOM_SEED", "42")),
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> Settings:
        """Load settings from a YAML mapping without mutating global state."""

        payload = _load_yaml_mapping(path)
        try:
            return cls(
                environment=Environment(payload.get("environment", Environment.LOCAL_PUBLIC.value)),
                data_root=Path(payload.get("data_root", "data")),
                allow_network=bool(payload.get("allow_network", False)),
                log_level=str(payload.get("log_level", "INFO")).upper(),
                random_seed=int(payload.get("random_seed", 42)),
            )
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Invalid settings in {Path(path).name}") from exc

    def assert_cohort_allowed(self, cohort: str) -> None:
        """Reject protected cohorts outside the approved secure environment."""

        if cohort.upper() in PROTECTED_COHORTS and self.environment is not Environment.SECURE_CBR:
            raise DataGovernanceError(
                f"Protected cohort {cohort!r} may only be accessed in environment={Environment.SECURE_CBR.value}"
            )


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"Invalid boolean value: {value!r}")


def _load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise ConfigurationError("Configuration root must be a mapping")
    return payload
