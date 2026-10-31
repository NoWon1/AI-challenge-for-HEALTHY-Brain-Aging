from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboards.streamlit_app import ImmutableRuntimeProxy


class DummyTarget:
    def __init__(self):
        self.value = 42
        self.config = {"key": "value"}
        self.df = pd.DataFrame({"a": [1, 2]})
        self.my_list = [1, 2, 3]
        self.my_set = {1, 2, 3}

        self.lst = [1, 2, 3]
        self.df = pd.DataFrame({"col": [1]})


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

    assert isinstance(proxy.df, pd.DataFrame)
    assert proxy.df is not target.df
    assert not isinstance(proxy.my_list, tuple)
    assert proxy.my_list == (1, 2, 3)
    assert not isinstance(proxy.my_set, frozenset)
    assert proxy.my_set == frozenset({1, 2, 3})

    # Test list tuple proxy
    assert proxy.lst == (1, 2, 3)

    # Test DataFrame proxy
    df_copy = proxy.df
    df_copy.loc[0, "col"] = 99
    assert proxy.df.loc[0, "col"] == 1  # Should not affect the original


def test_streamlit_judge_demo_smoke():
    app_path = Path(__file__).parents[1] / "dashboards" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=180)

    assert not app.exception
    assert len(app.tabs) == 3
    assert app.tabs[0].label == "Participant studio"
    assert app.tabs[1].label == "India-first validation"
    assert app.tabs[2].label == "Harmonisation audit"
    assert any("not for diagnosis" in error.value.lower() for error in app.error)
