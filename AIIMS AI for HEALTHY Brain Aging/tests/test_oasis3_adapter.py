import pandas as pd
import pytest

from neurosaarthi.core.errors import DataValidationError
from neurosaarthi.data.adapters.oasis3 import Oasis3Columns, Oasis3ManifestAdapter

COLUMNS = Oasis3Columns(
    participant_id="subject",
    visit_id="session",
    age_at_visit="age",
    time_from_baseline_days="days",
    image_uri="image",
    sequence="sequence",
    cognitive_instrument="instrument",
    cognitive_score="score",
)


def test_oasis_adapter_pseudonymizes_and_orders_visits(tmp_path):
    manifest = tmp_path / "sessions.csv"
    pd.DataFrame(
        {
            "subject": ["source-1", "source-1", "source-2"],
            "session": ["later", "baseline", "baseline"],
            "age": [71.0, 70.0, 66.0],
            "days": [365, 0, 0],
            "image": ["local/a.nii.gz", "local/b.nii.gz", "local/c.nii.gz"],
            "sequence": ["T1w", "T1w", "T1w"],
            "instrument": ["MMSE", "MMSE", "MMSE"],
            "score": [27, 28, 29],
        }
    ).to_csv(manifest, index=False)
    tables = Oasis3ManifestAdapter(manifest, COLUMNS, "a-secure-local-key").extract()
    assert len(tables.participants) == 2
    assert tables.visits["visit_index"].tolist() == [0, 1, 0]
    assert not tables.participants["participant_id_internal"].str.contains("source").any()
    assert tables.provenance["equivalence"].eq("EXACT").all()


def test_oasis_adapter_rejects_remote_images(tmp_path):
    manifest = tmp_path / "sessions.csv"
    pd.DataFrame(
        {
            "subject": ["source-1"],
            "session": ["baseline"],
            "age": [70.0],
            "days": [0],
            "image": ["https://example.test/scan.nii.gz"],
            "sequence": ["T1w"],
            "instrument": ["MMSE"],
            "score": [28],
        }
    ).to_csv(manifest, index=False)
    with pytest.raises(DataValidationError, match="Remote image"):
        Oasis3ManifestAdapter(manifest, COLUMNS, "a-secure-local-key").extract()
