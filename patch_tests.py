import re
with open("neurosaarthi-ad/tests/test_streamlit_app.py", "r") as f:
    content = f.read()

new_tests = """
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
"""

content = content.replace("    assert proxy.df.loc[0, \"col\"] == 1  # Should not affect the original", "    assert proxy.df.loc[0, \"col\"] == 1  # Should not affect the original\n" + new_tests)

target_add = """        self.lst = [1, 2, 3]
        self.df = pd.DataFrame({"col": [1]})
        import numpy as np
        self.arr = np.array([1, 2, 3])
        self.nested_tuple = ([1, 2], {"a": 3})
"""
content = content.replace("        self.lst = [1, 2, 3]\n        self.df = pd.DataFrame({\"col\": [1]})", target_add)

content = content.replace("from dashboards.streamlit_app import ImmutableRuntimeProxy", "from dashboards.streamlit_app import (\n    ImmutableListProxy,\n    ImmutableRuntimeProxy,\n    ImmutableSetProxy,\n)")
content = content.replace("assert isinstance(proxy.my_list, tuple)", "assert isinstance(proxy.my_list, (tuple, ImmutableListProxy))")
content = content.replace("assert isinstance(proxy.my_set, frozenset)", "assert isinstance(proxy.my_set, (frozenset, ImmutableSetProxy))")

with open("neurosaarthi-ad/tests/test_streamlit_app.py", "w") as f:
    f.write(content)
