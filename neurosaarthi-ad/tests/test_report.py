from evaluation.report import ValidationReport

def test_validation_report_to_markdown_no_notes():
    report = ValidationReport(
        title="Test Report",
        metrics={"accuracy": 0.95123, "loss": 0.1}
    )
    expected = (
        "# Test Report\n"
        "\n"
        "## Metrics\n"
        "- accuracy: 0.9512\n"
        "- loss: 0.1000\n"
    )
    assert report.to_markdown() == expected

def test_validation_report_to_markdown_with_notes():
    report = ValidationReport(
        title="Full Report",
        metrics={"c_index": 0.75},
        notes=["Model looks good.", "Needs more data."]
    )
    expected = (
        "# Full Report\n"
        "\n"
        "## Metrics\n"
        "- c_index: 0.7500\n"
        "\n"
        "## Notes\n"
        "- Model looks good.\n"
        "- Needs more data.\n"
    )
    assert report.to_markdown() == expected

def test_validation_report_to_markdown_empty_metrics_and_notes():
    report = ValidationReport(title="Empty Report")
    expected = (
        "# Empty Report\n"
        "\n"
        "## Metrics\n"
    )
    assert report.to_markdown() == expected
