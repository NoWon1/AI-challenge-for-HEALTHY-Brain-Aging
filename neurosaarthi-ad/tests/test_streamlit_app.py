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

def test_immutable_runtime_proxy():
    from dashboards.streamlit_app import ImmutableRuntimeProxy

    class MockTarget:
        def __init__(self):
            self.x = 10
            self.d = {"a": 1}

    target = MockTarget()
    proxy = ImmutableRuntimeProxy(target)

    # Test get behavior
    assert proxy.x == 10
    assert proxy.d["a"] == 1

    # Test mapping proxy on dicts
    import pytest
    with pytest.raises(TypeError):
        proxy.d["a"] = 2

    # Test mutation prevention
    with pytest.raises(AttributeError, match="Modification of 'x' forbidden"):
        proxy.x = 20

    with pytest.raises(AttributeError, match="Deletion of 'x' forbidden"):
        del proxy.x

    with pytest.raises(AttributeError, match="Modification of 'new_attr' forbidden"):
        proxy.new_attr = 100
