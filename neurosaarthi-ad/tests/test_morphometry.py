import numpy as np
import pytest

from neurosaarthi.core.errors import DataValidationError
from neurosaarthi.imaging.morphometry.volumes import compute_morphometry, longitudinal_volume_change


def test_morphometry_uses_physical_voxel_volume_and_icv_normalization():
    segmentation = np.zeros((10, 10, 10), dtype=np.uint8)
    segmentation[1:3, 1:3, 1:3] = 1  # 8 left voxels
    segmentation[4:7, 4:6, 4:6] = 2  # 12 right voxels
    segmentation[7:9, 7:9, 7:9] = 3  # 8 ventricular voxels
    icv = np.ones_like(segmentation)
    result = compute_morphometry(segmentation, (1.0, 1.0, 2.0), intracranial_mask=icv)
    assert result.left_hippocampal_volume_mm3 == 16.0
    assert result.right_hippocampal_volume_mm3 == 24.0
    assert result.total_hippocampal_volume_mm3 == 40.0
    assert result.hippocampal_asymmetry_index == pytest.approx(0.2)
    assert result.intracranial_volume_mm3 == 2000.0
    assert result.normalized_hippocampal_volume == pytest.approx(0.02)


def test_longitudinal_change_reports_positive_atrophy_for_volume_loss():
    result = longitudinal_volume_change(4000.0, 3800.0, 365.25)
    assert result.percentage_change == pytest.approx(-5.0)
    assert result.annualized_atrophy_percent == pytest.approx(5.0)


def test_morphometry_rejects_non_integral_labels():
    with pytest.raises(DataValidationError, match="finite integers"):
        compute_morphometry(np.full((3, 3, 3), 1.5), (1, 1, 1))
