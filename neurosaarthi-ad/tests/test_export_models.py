from pathlib import Path


def test_export_path_traversal_prevention(tmp_path):
    # This logic replicates the check inside scripts/export_models.py

    OUT_DIR = tmp_path
    resolved_out = OUT_DIR.resolve()

    malicious_inputs = ["../../etc/passwd", "../evil", "/root/artifact", "....//"]
    for bad_input in malicious_inputs:
        safe_modality = bad_input.replace("+", "").replace("/", "_").replace(" ", "_")
        target_path = (
            resolved_out / Path(f"risk_{safe_modality}.joblib").name
        ).resolve()

        assert target_path.is_relative_to(resolved_out)
        assert target_path.name.startswith("risk_")
