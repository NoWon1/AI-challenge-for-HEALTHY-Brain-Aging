"""Starter dashboard for NeuroSaarthi-AD."""

from __future__ import annotations


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Install the dashboard extra with: pip install -e .[dashboard]") from exc

    st.set_page_config(page_title="NeuroSaarthi-AD", layout="wide")
    st.title("NeuroSaarthi-AD Progression Studio")
    st.caption("Risk prediction, progression forecasting, and digital twin lite retrieval.")
    st.info("Wire cohort adapters and trained model artifacts here after data access is approved.")


if __name__ == "__main__":
    main()

