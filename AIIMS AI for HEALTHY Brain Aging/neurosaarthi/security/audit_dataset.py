"""Local metadata-only audit for unsafe participant-level exports.

The scanner inspects path names, delimited-file headers, JSON keys, and optional
DICOM header tag presence. It never uploads data, attempts re-identification,
or includes field values in its report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any

import yaml

from neurosaarthi.core.errors import ConfigurationError


class AuditMode(StrEnum):
    INTERNAL = "internal"
    EXPORT = "export"


class Severity(IntEnum):
    INFO = 10
    WARNING = 20
    BLOCK = 30


DIRECT_IDENTIFIER_FIELDS = frozenset(
    {
        "name",
        "patient_name",
        "patientname",
        "mrn",
        "medical_record_number",
        "accession_number",
        "accessionnumber",
        "phone",
        "email",
        "address",
        "aadhaar",
        "aadhar",
    }
)
QUASI_IDENTIFIER_FIELDS = frozenset(
    {
        "date_of_birth",
        "birth_date",
        "patientbirthdate",
        "visit_date",
        "scan_date",
        "exam_date",
        "participant_id",
        "participant_id_internal",
        "subject_id",
        "patient_id",
    }
)
PARTICIPANT_FILE_SUFFIXES = frozenset({".dcm", ".nii", ".nrrd", ".mha", ".mhd"})


@dataclass(frozen=True)
class AuditConfig:
    root: Path
    mode: AuditMode = AuditMode.EXPORT
    allowed_fields: frozenset[str] = frozenset()
    max_files: int = 100_000
    inspect_dicom_headers: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root).resolve())
        object.__setattr__(self, "allowed_fields", frozenset(item.lower() for item in self.allowed_fields))
        if self.max_files <= 0:
            raise ConfigurationError("max_files must be positive")


@dataclass(frozen=True)
class Finding:
    severity: Severity
    rule: str
    relative_path: str
    field: str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["severity"] = self.severity.name
        return payload


@dataclass
class AuditReport:
    root: str
    mode: AuditMode
    files_scanned: int = 0
    findings: list[Finding] = field(default_factory=list)

    @property
    def safe_for_export(self) -> bool:
        return not any(finding.severity >= Severity.BLOCK for finding in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "mode": self.mode.value,
            "files_scanned": self.files_scanned,
            "safe_for_export": self.safe_for_export,
            "summary": {
                level.name: sum(finding.severity is level for finding in self.findings) for level in Severity
            },
            "findings": [finding.to_dict() for finding in self.findings],
        }


def audit_dataset(config: AuditConfig) -> AuditReport:
    """Audit a local directory using metadata only.

    Symlinks are not followed. Results include relative paths and field names,
    never participant-level field values.
    """

    if not config.root.is_dir():
        raise FileNotFoundError(f"Audit root is not a directory: {config.root}")
    report = AuditReport(root="[LOCAL_AUDIT_ROOT]", mode=config.mode)
    for index, path in enumerate(_iter_files(config.root), start=1):
        if index > config.max_files:
            report.findings.append(
                Finding(
                    Severity.BLOCK, "scan_limit", ".", None, "File scan limit exceeded; audit is incomplete"
                )
            )
            break
        report.files_scanned += 1
        relative = path.relative_to(config.root).as_posix()
        path_reference = f"path#{hashlib.sha256(relative.encode('utf-8')).hexdigest()[:12]}"
        report.findings.extend(_audit_path(path, path_reference, config))
    return report


def _iter_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*"):
        if path.is_symlink():
            continue
        if path.is_file():
            yield path


def _audit_path(path: Path, relative: str, config: AuditConfig) -> list[Finding]:
    findings: list[Finding] = []
    lowered_parts = {part.lower() for part in path.parts}
    for identifier in DIRECT_IDENTIFIER_FIELDS:
        if any(identifier in part for part in lowered_parts):
            findings.append(
                Finding(
                    Severity.BLOCK,
                    "identifier_in_path",
                    relative,
                    None,
                    "Path may contain a direct identifier",
                )
            )
            break

    suffix = path.suffix.lower()
    compound_suffix = ".nii.gz" if path.name.lower().endswith(".nii.gz") else suffix
    if config.mode is AuditMode.EXPORT and compound_suffix in PARTICIPANT_FILE_SUFFIXES | {".nii.gz"}:
        findings.append(
            Finding(
                Severity.BLOCK,
                "participant_file",
                relative,
                None,
                "Participant-level imaging is not permitted in an export directory",
            )
        )

    fields: set[str] = set()
    try:
        if suffix in {".csv", ".tsv"}:
            fields = _read_delimited_header(path, "\t" if suffix == ".tsv" else ",")
        elif suffix == ".json":
            fields = _read_json_keys(path)
        elif suffix in {".yaml", ".yml"}:
            fields = _read_yaml_keys(path)
        elif suffix == ".dcm" and config.inspect_dicom_headers:
            fields = _read_dicom_fields(path)
    except (OSError, UnicodeError, csv.Error, json.JSONDecodeError, yaml.YAMLError) as exc:
        findings.append(
            Finding(
                Severity.WARNING,
                "metadata_unreadable",
                relative,
                None,
                f"Metadata inspection failed: {type(exc).__name__}",
            )
        )

    for field_name in sorted(fields):
        normalized = field_name.strip().lower().replace(" ", "_")
        if normalized in config.allowed_fields:
            continue
        if normalized in DIRECT_IDENTIFIER_FIELDS:
            findings.append(
                Finding(
                    Severity.BLOCK,
                    "direct_identifier_field",
                    relative,
                    field_name,
                    "Direct identifier field is present",
                )
            )
        elif normalized in QUASI_IDENTIFIER_FIELDS:
            severity = Severity.BLOCK if config.mode is AuditMode.EXPORT else Severity.WARNING
            findings.append(
                Finding(
                    severity,
                    "quasi_identifier_field",
                    relative,
                    field_name,
                    "Participant-level or exact-date field is present",
                )
            )
    return findings


def _read_delimited_header(path: Path, delimiter: str) -> set[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return {str(item) for item in next(csv.reader(stream, delimiter=delimiter), [])}


def _read_json_keys(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as stream:
        return set(_walk_keys(json.load(stream)))


def _read_yaml_keys(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as stream:
        return set(_walk_keys(yaml.safe_load(stream)))


def _walk_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key)
            yield from _walk_keys(nested)
    elif isinstance(value, list):
        for nested in value[:1000]:
            yield from _walk_keys(nested)


def _read_dicom_fields(path: Path) -> set[str]:
    try:
        import pydicom
    except ImportError:
        return set()
    dataset = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    sensitive_keywords = {"PatientName", "PatientID", "PatientBirthDate", "AccessionNumber"}
    return {keyword for keyword in sensitive_keywords if keyword in dataset}


def _config_from_yaml(path: Path, root_override: Path | None) -> AuditConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ConfigurationError("Audit configuration root must be a mapping")
    root = root_override or Path(payload.get("root", "."))
    if not root.is_absolute():
        root = path.resolve().parent / root
    return AuditConfig(
        root=root,
        mode=AuditMode(payload.get("mode", AuditMode.EXPORT.value)),
        allowed_fields=frozenset(payload.get("allowed_fields", [])),
        max_files=int(payload.get("max_files", 100_000)),
        inspect_dicom_headers=bool(payload.get("inspect_dicom_headers", True)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="YAML audit configuration")
    parser.add_argument("--root", type=Path, help="Directory to audit; overrides config root")
    parser.add_argument("--mode", choices=[mode.value for mode in AuditMode], default=AuditMode.EXPORT.value)
    parser.add_argument("--output", type=Path, help="Optional local JSON report path")
    args = parser.parse_args(argv)
    if args.config:
        config = _config_from_yaml(args.config, args.root)
    elif args.root:
        config = AuditConfig(root=args.root, mode=AuditMode(args.mode))
    else:
        parser.error("one of --config or --root is required")
    report = audit_dataset(config)
    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report.safe_for_export else 2


if __name__ == "__main__":
    raise SystemExit(main())
