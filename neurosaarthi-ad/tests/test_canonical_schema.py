import pytest

from neurosaarthi.core.errors import DataValidationError
from neurosaarthi.data.schemas.canonical import ImagingRecord, Visit


def test_visit_rejects_negative_longitudinal_time():
    with pytest.raises(DataValidationError, match="non-negative"):
        Visit("P-1", "V-1", 0, -1, 70.0)


def test_imaging_record_rejects_remote_uri():
    with pytest.raises(DataValidationError, match="Remote image URIs"):
        ImagingRecord("P-1", "V-1", "MRI", "T1w", "https://example.test/scan.nii.gz")
