from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from dashboards.streamlit_app import (
    ImmutableListProxy,
    ImmutableRuntimeProxy,
    ImmutableSetProxy,
)


class DummyTarget:
    def __init__(self):
        self.value = 42
        self.config = {"key": "value", "nested": {"inner": 1}}
        self.config = {"key": "value"}
        self.df = pd.DataFrame({"a": [1, 2]})
        self.my_list = [1, 2, 3]
        self.my_set = {1, 2, 3}

        self.lst = [1, 2, 3]
        self.df = pd.DataFrame({"col": [1]})
        self.arr = np.array([1, 2, 3])
        import numpy as np

        self.arr = np.array([1, 2, 3])
        self.nested_tuple = ([1, 2], {"a": 3})
        self.nested_tuple = ([1, 2], {"a": 1})



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

    # Test nested dictionary proxy
    with pytest.raises(TypeError):
        proxy.config["nested"]["inner"] = 2
    assert isinstance(proxy.df, pd.DataFrame)
    assert proxy.df is not target.df
    assert isinstance(proxy.my_list, (tuple, ImmutableListProxy))
    assert proxy.my_list == (1, 2, 3)
    assert isinstance(proxy.my_set, (frozenset, ImmutableSetProxy))
    assert proxy.my_set == frozenset({1, 2, 3})

    # Test list tuple proxy
    assert proxy.lst == (1, 2, 3)

    # Test list tuple proxy edge cases
    with pytest.raises(AttributeError):
        proxy.my_list.append(4)

    # Test numpy array proxy
    arr_copy = proxy.arr
    assert not arr_copy.flags.writeable
    with pytest.raises(ValueError, match="is read-only"):
        arr_copy[0] = 99
    # Test nested tuple proxy
    with pytest.raises(TypeError):
        proxy.nested_tuple[1]["a"] = 2
    assert proxy.nested_tuple[0] == [1, 2]

    # Test DataFrame proxy
    df_copy = proxy.df
    df_copy.loc[0, "col"] = 99
    assert proxy.df.loc[0, "col"] == 1  # Should not affect the original

    # List mutation
    with pytest.raises((TypeError, AttributeError)):
        proxy.my_list.append(4)
    with pytest.raises(TypeError):
        proxy.my_list[0] = 99

    # Set mutation
    with pytest.raises((TypeError, AttributeError)):
        proxy.my_set.add(4)

    # Nested tuple
    assert proxy.nested_tuple == ([1, 2], {"a": 3})
    with pytest.raises((TypeError, AttributeError)):
        proxy.nested_tuple[0].append(3)
    with pytest.raises(TypeError):
        proxy.nested_tuple[1]["a"] = 4

    # Numpy mutation
    assert proxy.arr.flags.writeable is False
    with pytest.raises(ValueError, match="assignment destination is read-only"):
        proxy.arr[0] = 99


def test_streamlit_judge_demo_smoke():
    app_path = Path(__file__).parents[1] / "dashboards" / "streamlit_app.py"
    app = AppTest.from_file(str(app_path)).run(timeout=600)

    assert not app.exception
    assert len(app.tabs) == 3
    assert app.tabs[0].label == "Participant studio"
    assert app.tabs[1].label == "India-first validation"
    assert app.tabs[2].label == "Harmonisation audit"
    assert any("not for diagnosis" in error.value.lower() for error in app.error)
