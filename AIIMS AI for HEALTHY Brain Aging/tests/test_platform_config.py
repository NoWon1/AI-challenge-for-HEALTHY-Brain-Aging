import pytest

from neurosaarthi.core.config import Environment, Settings
from neurosaarthi.core.errors import DataGovernanceError
from neurosaarthi.core.logging import redact_text


def test_secure_environment_rejects_network_access():
    with pytest.raises(DataGovernanceError, match="Network access"):
        Settings(environment=Environment.SECURE_CBR, allow_network=True)


def test_protected_cohort_rejected_outside_secure_environment():
    with pytest.raises(DataGovernanceError, match="secure_cbr"):
        Settings().assert_cohort_allowed("SANSCOG")


def test_identifier_values_are_redacted_from_logs():
    message = redact_text("participant_id=CBR-123 patientname:Example")
    assert "CBR-123" not in message
    assert "Example" not in message
    assert message.count("[REDACTED]") == 2
