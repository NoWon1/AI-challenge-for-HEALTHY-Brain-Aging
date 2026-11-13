import re
with open("neurosaarthi-ad/dashboards/streamlit_app.py", "r") as f:
    content = f.read()

# Replace the ImmutableRuntimeProxy definition to handle nested types
old_proxy = """class ImmutableRuntimeProxy:
    \"\"\"
    Read-only proxy wrapping DemoRuntime to prevent cross-session state corruption
    when cached via @st.cache_resource (CWE-374 / CWE-662).
    \"\"\"

    def __init__(self, target: Any) -> None:
        super().__setattr__("_target", target)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        # Protect internal dictionary lookups from in-place modification
        if isinstance(attr, dict):
            return MappingProxyType(attr)
        if isinstance(attr, pd.DataFrame):
            return attr.copy(deep=True)
        if isinstance(attr, list):
            return tuple(attr)
        if isinstance(attr, set):
            return frozenset(attr)
        return attr

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"Modification of attribute '{name}' is forbidden: cached runtime is read-only across sessions."
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"Deletion of attribute '{name}' is forbidden: cached runtime is read-only across sessions."
        )"""

new_proxy = """from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet

def _secure_value(val):
    if isinstance(val, dict):
        return ImmutableDictProxy(val)
    if isinstance(val, list):
        return ImmutableListProxy(val)
    if isinstance(val, tuple):
        if hasattr(type(val), "_fields"):
            return type(val)(*(_secure_value(x) for x in val))
        return tuple(_secure_value(x) for x in val)
    if isinstance(val, set):
        return ImmutableSetProxy(val)
    if isinstance(val, pd.DataFrame):
        return val.copy(deep=True)
    if isinstance(val, np.ndarray):
        val_copy = val.copy()
        val_copy.flags.writeable = False
        return val_copy
    return val

class ImmutableDictProxy(Mapping):
    def __init__(self, d):
        self._d = d

    def __getitem__(self, key):
        return _secure_value(self._d[key])

    def __iter__(self):
        return iter(self._d)

    def __len__(self):
        return len(self._d)

class ImmutableListProxy(Sequence):
    def __init__(self, lst):
        self._lst = lst

    def __getitem__(self, idx):
        return _secure_value(self._lst[idx])

    def __len__(self):
        return len(self._lst)

    def __eq__(self, other):
        if not isinstance(other, (list, tuple, Sequence)):
            return False
        if len(self) != len(other):
            return False
        return all(a == b for a, b in zip(self, other))

class ImmutableSetProxy(AbstractSet):
    def __init__(self, s):
        self._s = s

    def __contains__(self, item):
        return item in self._s

    def __iter__(self):
        return (_secure_value(x) for x in self._s)

    def __len__(self):
        return len(self._s)

    def __eq__(self, other):
        if not isinstance(other, (set, frozenset, AbstractSet)):
            return False
        return set(self) == set(other)

class ImmutableRuntimeProxy:
    \"\"\"
    Read-only proxy wrapping DemoRuntime to prevent cross-session state corruption
    when cached via @st.cache_resource (CWE-374 / CWE-662).
    \"\"\"

    def __init__(self, target: Any) -> None:
        super().__setattr__("_target", target)

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._target, name)
        return _secure_value(attr)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"Modification of attribute '{name}' is forbidden: cached runtime is read-only across sessions."
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError(
            f"Deletion of attribute '{name}' is forbidden: cached runtime is read-only across sessions."
        )"""

content = content.replace(old_proxy, new_proxy)
with open("neurosaarthi-ad/dashboards/streamlit_app.py", "w") as f:
    f.write(content)
