import json

import numpy as np
import pytest

from neurosaarthi.cli import main

nib = pytest.importorskip("nibabel")


def _synthetic_t1(shape=(48, 48, 48)):
    coordinates = np.indices(shape)
    center = (np.asarray(shape) - 1)[:, None, None, None] / 2
    radius = np.sqrt(np.sum(((coordinates - center) / 15.0) ** 2, axis=0))
    return np.where(radius <= 1.0, 100.0 + 20.0 * (1.0 - radius), 0.0).astype(np.float32)


def test_synthetic_nifti_qc_preprocess_and_morphometry_cli(tmp_path, capsys):
    source = tmp_path / "synthetic_T1w.nii.gz"
    processed = tmp_path / "synthetic_T1w_preprocessed.nii.gz"
    qc_report = tmp_path / "qc.json"
    nib.save(nib.Nifti1Image(_synthetic_t1(), np.eye(4)), source)

    assert main(["imaging", "qc", "--image", str(source)]) == 0
    qc_payload = json.loads(capsys.readouterr().out)
    assert qc_payload["passed"] is True

    config = tmp_path / "preprocess.yaml"
    config.write_text(
        "\n".join(
            [
                f"input_image: '{source.as_posix()}'",
                f"output_image: '{processed.as_posix()}'",
                f"qc_report: '{qc_report.as_posix()}'",
                "preprocessing:",
                "  target_spacing_mm: [1.0, 1.0, 1.0]",
                "  output_shape: [40, 40, 40]",
                "  use_n4_bias_correction: false",
                "  fail_on_qc_error: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert main(["imaging", "preprocess", "--config", str(config)]) == 0
    preprocess_payload = json.loads(capsys.readouterr().out)
    assert preprocess_payload["output_shape"] == [40, 40, 40]
    assert processed.is_file() and qc_report.is_file()

    segmentation = np.zeros((20, 20, 20), dtype=np.uint8)
    segmentation[1:4, 1:4, 1:4] = 1
    segmentation[5:8, 5:8, 5:8] = 2
    segmentation[10:12, 10:12, 10:12] = 3
    segmentation_path = tmp_path / "synthetic_labels.nii.gz"
    intracranial_path = tmp_path / "synthetic_icv.nii.gz"
    nib.save(nib.Nifti1Image(segmentation, np.eye(4)), segmentation_path)
    nib.save(nib.Nifti1Image(np.ones_like(segmentation), np.eye(4)), intracranial_path)

    assert (
        main(
            [
                "features",
                "morphometry",
                "--segmentation",
                str(segmentation_path),
                "--intracranial-mask",
                str(intracranial_path),
            ]
        )
        == 0
    )
    morphometry = json.loads(capsys.readouterr().out)
    assert morphometry["total_hippocampal_volume_mm3"] == 54.0
    assert morphometry["intracranial_volume_mm3"] == 8000.0
