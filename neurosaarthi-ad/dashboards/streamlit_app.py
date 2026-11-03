"""Judge-facing Streamlit prototype for NeuroSaarthi-AD."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


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
    """
    Read-only proxy wrapping DemoRuntime to prevent cross-session state corruption
    when cached via @st.cache_resource (CWE-374 / CWE-662).
    """

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
        )


COLORS = {
    "teal": "#0F5C5B",
    "teal_dark": "#173B3A",
    "saffron": "#D98E32",
    "coral": "#C96755",
    "cream": "#F6F3EA",
    "paper": "#FFFDF8",
    "sage": "#DCE9E3",
    "muted": "#657775",
}


def _inject_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {COLORS["cream"]}; color: {COLORS["teal_dark"]}; }}
        [data-testid="stSidebar"] {{ background: #EAF0EB; border-right: 1px solid #CBD8D2; }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: {COLORS["teal_dark"]}; }}
        .block-container {{ max-width: 1320px; padding-top: 1.8rem; padding-bottom: 3rem; }}
        h1, h2, h3 {{ color: {COLORS["teal_dark"]}; letter-spacing: -0.02em; }}
        .ns-hero {{
            background: linear-gradient(125deg, #123F3E 0%, #0F5C5B 58%, #38736C 100%);
            border-radius: 22px; padding: 2rem 2.1rem; color: white; margin-bottom: 1rem;
            box-shadow: 0 16px 40px rgba(28, 61, 58, 0.16);
        }}
        .ns-kicker {{ color: #F2C985; font-size: 0.78rem; font-weight: 800; letter-spacing: 0.12em; text-transform: uppercase; }}
        .ns-hero h1 {{ color: white; margin: 0.3rem 0 0.55rem; font-size: clamp(2rem, 4vw, 3.25rem); line-height: 1.04; }}
        .ns-hero p {{ color: #E1ECE8; max-width: 780px; font-size: 1.02rem; margin: 0; }}
        .ns-pills {{ display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 1.2rem; }}
        .ns-pill {{ background: rgba(255,255,255,.11); border: 1px solid rgba(255,255,255,.19); border-radius: 999px; padding: .35rem .7rem; font-size: .78rem; }}
        .ns-risk {{ background: {COLORS["paper"]}; border: 1px solid #D9E2DD; border-radius: 16px; padding: 1rem 1.05rem; min-height: 142px; box-shadow: 0 8px 22px rgba(29,58,55,.06); }}
        .ns-risk .label {{ color: {COLORS["muted"]}; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 750; }}
        .ns-risk .value {{ color: {COLORS["teal"]}; font-size: 2.2rem; font-weight: 780; line-height: 1.15; margin: .35rem 0; }}
        .ns-risk .interval {{ color: {COLORS["muted"]}; font-size: .82rem; }}
        .ns-section-note {{ color: {COLORS["muted"]}; margin-top: -.55rem; margin-bottom: 1rem; }}
        .ns-callout {{ background: #FFF5E5; border: 1px solid #EBC994; border-left: 5px solid {COLORS["saffron"]}; border-radius: 10px; padding: .75rem 1rem; color: #604A29; margin: .8rem 0 1rem; }}
        .ns-flow {{ display:grid; grid-template-columns: repeat(4,1fr); gap:.65rem; margin: 1rem 0; }}
        .ns-flow div {{ background:{COLORS["paper"]}; border:1px solid #D9E2DD; border-radius:12px; padding:.85rem; font-size:.85rem; }}
        .ns-flow b {{ color:{COLORS["teal"]}; display:block; margin-bottom:.25rem; }}
        div[data-testid="stMetric"] {{ background: {COLORS["paper"]}; border: 1px solid #D9E2DD; padding: .8rem 1rem; border-radius: 14px; }}
        div[data-testid="stDataFrame"] {{ border: 1px solid #D9E2DD; border-radius: 12px; overflow: hidden; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
        .stTabs [data-baseweb="tab"] {{ background: #E2EBE6; border-radius: 999px; padding: .4rem 1rem; }}
        .stTabs [aria-selected="true"] {{ background: {COLORS["teal"]} !important; color: white !important; }}
        @media (max-width: 800px) {{ .ns-flow {{ grid-template-columns: 1fr 1fr; }} .block-container {{ padding-left: 1rem; padding-right: 1rem; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )


from collections.abc import Mapping, Sequence
from collections.abc import Set as AbstractSet


def _secure_value(val):
    if isinstance(val, dict):
        return ImmutableDictProxy(val)
    if isinstance(val, list):
        return ImmutableListProxy(val)
    if isinstance(val, tuple):
        return tuple(_secure_value(x) for x in val)
    if isinstance(val, set):
        return ImmutableSetProxy(val)
    if isinstance(val, tuple):
        if hasattr(type(val), "_fields"):
            return type(val)(*(_secure_value(x) for x in val))
        return tuple(_secure_value(x) for x in val)
    if isinstance(val, pd.DataFrame):
        return val.copy(deep=True)
    if isinstance(val, np.ndarray):
        val_copy = val.copy()
        val_copy.flags.writeable = False
        return val_copy
    return val
