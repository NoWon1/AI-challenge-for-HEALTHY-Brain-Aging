import numpy as np
import pytest

from neurosaarthi.core.errors import DataValidationError
from neurosaarthi.imaging.io.nifti import ImageVolume
from neurosaarthi.imaging.preprocessing.t1 import T1PreprocessConfig, T1Preprocessor
from neurosaarthi.imaging.qc.structural import run_structural_qc


def _synthetic_t1(shape=(48, 48, 48)):
    coordinates = np.indices(shape)
    center = (np.asarray(shape) - 1)[:, None, None, None] / 2
    radius = np.sqrt(np.sum(((coordinates - center) / 15.0) ** 2, axis=0))
    data = np.where(radius <= 1.0, 100.0 + 20.0 * (1.0 - radius), 0.0).astype(np.float32)
    return ImageVolume(data=data, affine=np.eye(4))


def test_structural_qc_passes_well_formed_synthetic_volume():
    report = run_structural_qc(_synthetic_t1())
    assert report.passed
    assert report.finite_fraction == 1.0
    assert report.foreground_fraction > 0


def test_structural_qc_rejects_non_finite_voxel():
    volume = _synthetic_t1()
    data = volume.data.copy()
    data[20, 20, 20] = np.nan
    report = run_structural_qc(ImageVolume(data, np.eye(4)))
    assert not report.passed
    assert "non_finite" in {flag.code for flag in report.flags}


def test_preprocessing_tracks_crop_and_inverts_discrete_mask():
    volume = _synthetic_t1()
    config = T1PreprocessConfig(target_spacing_mm=(1, 1, 1), output_shape=(40, 40, 40))
    result = T1Preprocessor(config).preprocess(volume)
    assert result.image.shape == (40, 40, 40)
    assert np.isclose(result.image[result.foreground_mask].mean(), 0.0, atol=1e-5)
    processed_mask = np.zeros(result.image.shape, dtype=np.uint8)
    processed_mask[18:22, 18:22, 18:22] = 2
    restored = result.transform.inverse_mask(processed_mask)
    assert restored.shape == volume.data.shape
    assert set(np.unique(restored)) <= {0, 2}


def test_preprocessing_rejects_constant_image():
    volume = ImageVolume(np.ones((48, 48, 48), dtype=np.float32), np.eye(4))
    with pytest.raises(DataValidationError, match="failed input QC"):
        T1Preprocessor(T1PreprocessConfig(output_shape=(48, 48, 48))).preprocess(volume)
