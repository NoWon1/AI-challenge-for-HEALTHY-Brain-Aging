import json

from neurosaarthi.security.audit_dataset import AuditConfig, AuditMode, Severity, audit_dataset


def test_export_audit_blocks_identifiers_without_reading_values(tmp_path):
    (tmp_path / "rows.csv").write_text("participant_id,age,metric\nSYNTHETIC-001,70,0.5\n", encoding="utf-8")
    report = audit_dataset(AuditConfig(root=tmp_path, mode=AuditMode.EXPORT))
    assert not report.safe_for_export
    assert any(finding.rule == "quasi_identifier_field" for finding in report.findings)
    rendered = json.dumps(report.to_dict())
    assert "SYNTHETIC-001" not in rendered


def test_export_audit_allows_aggregate_schema(tmp_path):
    (tmp_path / "metrics.csv").write_text("subgroup,metric,estimate\nage_65_74,mae,4.2\n", encoding="utf-8")
    report = audit_dataset(AuditConfig(root=tmp_path, mode=AuditMode.EXPORT))
    assert report.safe_for_export
    assert not any(finding.severity is Severity.BLOCK for finding in report.findings)


def test_export_audit_blocks_participant_imaging(tmp_path):
    (tmp_path / "scan.nii.gz").write_bytes(b"not-a-real-nifti")
    report = audit_dataset(AuditConfig(root=tmp_path, mode=AuditMode.EXPORT))
    assert any(finding.rule == "participant_file" for finding in report.findings)
