from data_contracts.schema import load_contract, validate_columns


def test_common_contract_loads():
    contracts = load_contract()
    assert "participants" in contracts
    assert contracts["participants"].primary_key == "participant_id"


def test_contract_rejects_missing_required_column():
    try:
        validate_columns("participants", ["participant_id"])
    except ValueError as exc:
        assert "cohort" in str(exc)
    else:
        raise AssertionError("Expected missing cohort column to fail validation")
