from pathlib import Path

from dashboards.streamlit_app import _sanitize_markdown, ImmutableRuntimeProxy
from streamlit.testing.v1 import AppTest
import pytest

class DummyTarget:
    def __init__(self):
        self.value = 42
        self.config = {"key": "value"}

def test_immutable_runtime_proxy():
    target = DummyTarget()
    proxy = ImmutableRuntimeProxy(target)

    assert proxy.value == 42
    assert proxy.config["key"] == "value"

    with pytest.raises(AttributeError, match="Modification of attribute"):
        proxy.value = 100

    with pytest.raises(AttributeError, match="Deletion of attribute"):
        del proxy.value

    # Test dictionary mapping proxy
    with pytest.raises(TypeError):
        proxy.config["key"] = "new_value"

def test_streamlit_judge_demo_smoke():
    app_path = Path(__file__).parents[1] / "dashboards" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=45)

    assert not app.exception
    assert len(app.tabs) == 3
    assert app.tabs[0].label == "Participant studio"
    assert app.tabs[1].label == "India-first validation"
    assert app.tabs[2].label == "Harmonisation audit"
    assert any("not for diagnosis" in error.value.lower() for error in app.error)
