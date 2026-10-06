import pytest
from dashboards.streamlit_app import _load_runtime, ReadOnlyFacade
from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_judge_demo_smoke():
    app_path = Path(__file__).parents[1] / "dashboards" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=45)

    assert not app.exception
    assert len(app.tabs) == 3
    assert app.tabs[0].label == "Participant studio"
    assert app.tabs[1].label == "India-first validation"
    assert app.tabs[2].label == "Harmonisation audit"
    assert any("not for diagnosis" in error.value.lower() for error in app.error)


def test_cached_runtime_is_read_only():
    """Verify that the cached runtime is wrapped in a ReadOnlyFacade to prevent state corruption."""
    runtime = _load_runtime()

    assert isinstance(runtime, ReadOnlyFacade)

    with pytest.raises(TypeError, match="Modification of attribute 'seed' is forbidden: cached runtime is read-only across sessions."):
        runtime.seed = 999

    with pytest.raises(TypeError, match="Deletion of attribute 'seed' is forbidden: cached runtime is read-only across sessions."):
        del runtime.seed
