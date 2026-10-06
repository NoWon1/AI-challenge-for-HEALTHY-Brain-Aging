from pathlib import Path

import pytest
from types import MappingProxyType

from streamlit.testing.v1 import AppTest
from dashboards.streamlit_app import _load_runtime


def test_cached_runtime_is_immutable():
    """Verify that DemoRuntime wrapped by ImmutableRuntimeProxy is read-only."""
    runtime_proxy = _load_runtime()

    # Assert setting attributes is blocked
    with pytest.raises(AttributeError, match="Modification of attribute"):
        runtime_proxy.some_new_attribute = "test"

    # Assert deleting attributes is blocked
    with pytest.raises(AttributeError, match="Deletion of attribute"):
        del runtime_proxy.bundle

    # Assert dictionaries are wrapped in MappingProxyType
    assert isinstance(runtime_proxy.modality_reference, MappingProxyType)


def test_streamlit_judge_demo_smoke():
    app_path = Path(__file__).parents[1] / "dashboards" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=45)

    assert not app.exception
    assert len(app.tabs) == 3
    assert app.tabs[0].label == "Participant studio"
    assert app.tabs[1].label == "India-first validation"
    assert app.tabs[2].label == "Harmonisation audit"
    assert any("not for diagnosis" in error.value.lower() for error in app.error)
