"""Command-line interface for local, research-only workflows."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from neurosaarthi import __version__
from neurosaarthi.core.errors import NeuroSaarthiError
from neurosaarthi.security.audit_dataset import AuditMode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurosaarthi", description=__doc__)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="Dataset adapter and contract operations")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    validate = data_commands.add_parser("validate", help="Validate a configured local dataset")
    validate.add_argument("--dataset", required=True, choices=["oasis3"])
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--secret-env", default="NEUROSAARTHI_PSEUDONYM_KEY")
    validate.set_defaults(handler=_data_validate)

    imaging = commands.add_parser("imaging", help="MRI ingestion, QC, and preprocessing")
    imaging_commands = imaging.add_subparsers(dest="imaging_command", required=True)
    qc = imaging_commands.add_parser("qc", help="Run structural MRI QC")
    qc.add_argument("--image", type=Path, required=True)
    qc.set_defaults(handler=_imaging_qc)
    preprocess = imaging_commands.add_parser("preprocess", help="Run configured T1 preprocessing")
    preprocess.add_argument("--config", type=Path, required=True)
    preprocess.set_defaults(handler=_imaging_preprocess)

    features = commands.add_parser("features", help="Quantitative imaging biomarkers")
    feature_commands = features.add_subparsers(dest="feature_command", required=True)
    morphometry = feature_commands.add_parser("morphometry", help="Calculate volumes from a local mask")
    morphometry.add_argument("--segmentation", type=Path, required=True)
    morphometry.add_argument("--intracranial-mask", type=Path)
    morphometry.add_argument("--labels-config", type=Path)
    morphometry.add_argument("--output", type=Path)
    morphometry.set_defaults(handler=_features_morphometry)

    train = commands.add_parser("train", help="Leakage-safe baseline experiments")
    train_commands = train.add_subparsers(dest="train_command", required=True)
    brain_age = train_commands.add_parser("brain-age", help="Train/calibrate/test elastic-net brain age")
    brain_age.add_argument("--config", type=Path, required=True)
    brain_age.set_defaults(handler=_train_brain_age)

    security = commands.add_parser("security", help="Local governance safeguards")
    security_commands = security.add_subparsers(dest="security_command", required=True)
    audit = security_commands.add_parser("audit-dataset", help="Audit a directory before export")
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--mode", choices=[mode.value for mode in AuditMode], default=AuditMode.EXPORT.value)
    audit.add_argument("--output", type=Path)
    audit.set_defaults(handler=_security_audit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (NeuroSaarthiError, FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


def _data_validate(args: argparse.Namespace) -> int:
    from neurosaarthi.data.adapters.oasis3 import Oasis3ManifestAdapter

    secret = os.getenv(args.secret_env)
    if not secret:
        raise ValueError(f"Set {args.secret_env} in the local secure environment")
    adapter = Oasis3ManifestAdapter.from_yaml(args.config, secret)
    tables = adapter.extract()
    report = {
        "dataset": "OASIS-3",
        "valid": True,
        "counts": {
            "participants": len(tables.participants),
            "visits": len(tables.visits),
            "imaging_records": len(tables.imaging),
            "cognitive_observations": len(tables.cognitive),
        },
        "participant_ids_emitted": False,
        "network_used": False,
    }
    _print_json(report)
    return 0


def _imaging_qc(args: argparse.Namespace) -> int:
    from neurosaarthi.imaging.io.nifti import load_nifti
    from neurosaarthi.imaging.qc.structural import run_structural_qc

    report = run_structural_qc(load_nifti(args.image))
    _print_json(report.to_dict())
    return 0 if report.passed else 2


def _imaging_preprocess(args: argparse.Namespace) -> int:
    from neurosaarthi.imaging.io.nifti import load_nifti, save_nifti
    from neurosaarthi.imaging.preprocessing.t1 import T1PreprocessConfig, T1Preprocessor

    payload = _load_yaml(args.config)
    base = args.config.resolve().parent
    input_path = _local_path(payload, "input_image", base)
    output_path = _local_path(payload, "output_image", base)
    config_payload = payload.get("preprocessing", {})
    if not isinstance(config_payload, dict):
        raise ValueError("preprocessing must be a mapping")
    config = T1PreprocessConfig(
        target_spacing_mm=_triple(config_payload.get("target_spacing_mm", [1.0, 1.0, 1.0]), float),
        output_shape=_triple(config_payload.get("output_shape", [160, 192, 160]), int),
        intensity_lower_percentile=float(config_payload.get("intensity_lower_percentile", 0.5)),
        intensity_upper_percentile=float(config_payload.get("intensity_upper_percentile", 99.5)),
        use_n4_bias_correction=bool(config_payload.get("use_n4_bias_correction", False)),
        fail_on_qc_error=bool(config_payload.get("fail_on_qc_error", True)),
    )
    result = T1Preprocessor(config).preprocess(load_nifti(input_path))
    save_nifti(result.image, result.affine, output_path)
    report = {
        "output_shape": list(result.image.shape),
        "output_spacing_mm": list(config.target_spacing_mm),
        "input_qc": result.input_qc.to_dict(),
        "output_qc": result.output_qc.to_dict(),
        "output_path": str(output_path),
    }
    report_path_text = payload.get("qc_report")
    if report_path_text:
        report_path = _resolve_local_path(report_path_text, base)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_json(report)
    return 0


def _features_morphometry(args: argparse.Namespace) -> int:
    from neurosaarthi.imaging.io.nifti import load_nifti
    from neurosaarthi.imaging.morphometry.volumes import AnatomyLabels, compute_morphometry

    segmentation = load_nifti(args.segmentation, canonical=False)
    intracranial = load_nifti(args.intracranial_mask, canonical=False) if args.intracranial_mask else None
    labels = AnatomyLabels()
    if args.labels_config:
        payload = _load_yaml(args.labels_config)
        labels = AnatomyLabels(
            left_hippocampus=tuple(int(item) for item in payload.get("left_hippocampus", [1])),
            right_hippocampus=tuple(int(item) for item in payload.get("right_hippocampus", [2])),
            ventricles=tuple(int(item) for item in payload.get("ventricles", [3])),
            brain_tissue=tuple(int(item) for item in payload.get("brain_tissue", [])),
        )
    report = compute_morphometry(
        segmentation.data,
        segmentation.voxel_spacing_mm,
        labels=labels,
        intracranial_mask=intracranial.data if intracranial else None,
    ).to_dict()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_json(report)
    return 0


def _train_brain_age(args: argparse.Namespace) -> int:
    import pandas as pd

    from neurosaarthi.training.brain_age import run_brain_age_experiment

    payload = _load_yaml(args.config)
    base = args.config.resolve().parent
    feature_table = _local_path(payload, "feature_table", base)
    frame = pd.read_csv(feature_table, low_memory=False)
    features = payload.get("feature_columns")
    if not isinstance(features, list) or not features:
        raise ValueError("feature_columns must be a non-empty list")
    result = run_brain_age_experiment(
        frame,
        feature_columns=[str(item) for item in features],
        age_col=str(payload.get("age_column", "age_at_visit")),
        participant_col=str(payload.get("participant_column", "participant_id_internal")),
        test_size=float(payload.get("test_size", 0.20)),
        calibration_size=float(payload.get("calibration_size", 0.20)),
        seed=int(payload.get("seed", 42)),
    )
    report = result.aggregate_report()
    output_text = payload.get("aggregate_report")
    if output_text:
        output = _resolve_local_path(output_text, base)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_json(report)
    return 0


def _security_audit(args: argparse.Namespace) -> int:
    from neurosaarthi.security.audit_dataset import AuditConfig, audit_dataset

    report = audit_dataset(AuditConfig(root=args.root, mode=AuditMode(args.mode)))
    payload = report.to_dict()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_json(payload)
    return 0 if report.safe_for_export else 2


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("Configuration root must be a mapping")
    return payload


def _local_path(payload: dict[str, Any], key: str, base: Path) -> Path:
    if key not in payload:
        raise ValueError(f"Missing configuration key: {key}")
    return _resolve_local_path(payload[key], base)


def _resolve_local_path(value: Any, base: Path) -> Path:
    text = str(value)
    if text.lower().startswith(("http://", "https://", "s3://", "gs://")):
        raise ValueError("Remote paths are disabled")
    path = Path(text)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _triple(value: Any, converter: type) -> tuple[Any, Any, Any]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise ValueError("Expected a three-item sequence")
    return converter(value[0]), converter(value[1]), converter(value[2])


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    raise SystemExit(main())
