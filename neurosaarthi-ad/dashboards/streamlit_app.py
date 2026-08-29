"""Judge-facing Streamlit prototype for NeuroSaarthi-AD."""

from __future__ import annotations

import sys
from pathlib import Path
import html

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import html
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from demo.runtime import (
    PRESET_PROFILES,
    ParticipantProfile,
    build_demo_runtime,
    profile_with,
)
from demo.synthetic import generate_demo_cohort


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


@st.cache_resource(show_spinner="Preparing the synthetic seven-cohort workbench…")
def _load_runtime():
    return build_demo_runtime(
        generate_demo_cohort(seed=42, n_per_cohort=120), n_bootstrap=12
    )


def _optional_slider(
    label: str,
    enabled: bool,
    default: float | None,
    minimum: float,
    maximum: float,
    step: float,
    key: str,
):
    if not enabled:
        return None
    fallback = (minimum + maximum) / 2
    value = float(default) if default is not None else fallback
    return st.sidebar.slider(label, minimum, maximum, value, step, key=key)


def _profile_controls() -> ParticipantProfile:
    st.sidebar.markdown("## Synthetic participant")
    st.sidebar.caption(
        "Choose a fictional case, then edit the evidence available to the model."
    )
    preset_name = st.sidebar.selectbox(
        "Starting case", list(PRESET_PROFILES), key="preset_case"
    )
    base = PRESET_PROFILES[preset_name]
    prefix = base.participant_id

    age = st.sidebar.slider("Age", 48, 90, int(base.age), 1, key=f"{prefix}-age")
    sex = st.sidebar.selectbox(
        "Sex",
        ["Female", "Male"],
        index=0 if base.sex == "Female" else 1,
        key=f"{prefix}-sex",
    )
    education = st.sidebar.slider(
        "Education (years)",
        0,
        22,
        int(base.education_years),
        1,
        key=f"{prefix}-education",
    )
    setting = st.sidebar.selectbox(
        "Community setting",
        ["urban", "rural"],
        index=0 if base.urban_rural == "urban" else 1,
        key=f"{prefix}-setting",
    )

    with st.sidebar.expander("Cognition", expanded=True):
        cognition = st.slider(
            "Cognitive composite (0–30)",
            10.0,
            30.0,
            float(base.cognitive_score),
            0.1,
            key=f"{prefix}-cognition",
        )
        memory = st.slider(
            "Memory z-score",
            -3.0,
            2.5,
            float(base.memory_score),
            0.05,
            key=f"{prefix}-memory",
        )
        executive = st.slider(
            "Executive z-score",
            -3.0,
            2.5,
            float(base.executive_score),
            0.05,
            key=f"{prefix}-executive",
        )

    with st.sidebar.expander("MRI"):
        mri_available = st.checkbox(
            "MRI features available",
            value=base.hippocampal_volume_mm3 is not None,
            key=f"{prefix}-mri-on",
        )
        hippocampal = _optional_slider(
            "Hippocampal volume (mm³)",
            mri_available,
            base.hippocampal_volume_mm3,
            3500.0,
            9000.0,
            50.0,
            f"{prefix}-hippo",
        )
        wmh = _optional_slider(
            "WMH burden (mL)",
            mri_available,
            base.wmh_burden_ml,
            0.0,
            18.0,
            0.1,
            f"{prefix}-wmh",
        )

    with st.sidebar.expander("Blood biochemistry"):
        blood_available = st.checkbox(
            "Blood features available",
            value=base.hba1c_percent is not None,
            key=f"{prefix}-blood-on",
        )
        hba1c = _optional_slider(
            "HbA1c (%)",
            blood_available,
            base.hba1c_percent,
            4.0,
            9.0,
            0.1,
            f"{prefix}-hba1c",
        )
        crp = _optional_slider(
            "hs-CRP (mg/L)",
            blood_available,
            base.hs_crp_mg_l,
            0.1,
            12.0,
            0.1,
            f"{prefix}-crp",
        )

    with st.sidebar.expander("OCT / OCTA"):
        oct_available = st.checkbox(
            "Retinal features available",
            value=base.rnfl_um is not None,
            key=f"{prefix}-oct-on",
        )
        rnfl = _optional_slider(
            "RNFL thickness (µm)",
            oct_available,
            base.rnfl_um,
            60.0,
            112.0,
            0.5,
            f"{prefix}-rnfl",
        )
        vessel = _optional_slider(
            "Vessel density (%)",
            oct_available,
            base.vessel_density_percent,
            34.0,
            58.0,
            0.5,
            f"{prefix}-vessel",
        )

    with st.sidebar.expander("Genomics · GenomeIndia-aware context"):
        genomics_available = st.checkbox(
            "Genomic features available",
            value=base.apoe_e4_count is not None,
            key=f"{prefix}-genomics-on",
        )
        if genomics_available:
            apoe = float(
                st.select_slider(
                    "APOE ε4 allele count",
                    options=[0, 1, 2],
                    value=int(base.apoe_e4_count or 0),
                    key=f"{prefix}-apoe",
                )
            )
            ancestry = st.slider(
                "Synthetic ancestry PC1",
                -1.0,
                2.0,
                float(base.ancestry_pc1 or 1.0),
                0.05,
                key=f"{prefix}-ancestry",
            )
        else:
            apoe = None
            ancestry = None

    return profile_with(
        base,
        age=float(age),
        sex=sex,
        education_years=float(education),
        urban_rural=setting,
        cognitive_score=float(cognition),
        memory_score=float(memory),
        executive_score=float(executive),
        hippocampal_volume_mm3=hippocampal,
        wmh_burden_ml=wmh,
        hba1c_percent=hba1c,
        hs_crp_mg_l=crp,
        rnfl_um=rnfl,
        vessel_density_percent=vessel,
        apoe_e4_count=apoe,
        ancestry_pc1=ancestry,
    )


def _risk_card(row: dict) -> str:
    horizon = html.escape(str(int(row['horizon'])))
    risk = html.escape(f"{row['risk']:.0%}")
    lower = html.escape(f"{row['lower']:.0%}")
    upper = html.escape(f"{row['upper']:.0%}")
    return f"""
    <div class="ns-risk">
      <div class="label">{horizon}-year progression risk</div>
      <div class="value">{risk}</div>
      <div class="interval">80% synthetic interval · {lower}–{upper}</div>
    </div>
    """


def _trajectory_chart(frame: pd.DataFrame) -> alt.Chart:
    base = alt.Chart(frame).encode(
        x=alt.X(
            "year:Q", title="Years from this visit", scale=alt.Scale(domain=[0, 5])
        ),
    )
    band = base.mark_area(color=COLORS["sage"], opacity=0.8).encode(
        y=alt.Y(
            "lower:Q", title="Cognitive composite", scale=alt.Scale(domain=[0, 30])
        ),
        y2="upper:Q",
    )
    line = base.mark_line(
        color=COLORS["teal"],
        strokeWidth=3,
        point=alt.OverlayMarkDef(filled=True, size=65),
    ).encode(
        y="predicted_score:Q",
        tooltip=[
            alt.Tooltip("year:Q", title="Year"),
            alt.Tooltip("predicted_score:Q", title="Projected score", format=".1f"),
        ],
    )
    return (band + line).properties(height=300)


def _driver_chart(frame: pd.DataFrame) -> alt.Chart:
    plot = frame.copy()
    plot["effect_points"] = plot["relative_effect"] * 100
    return (
        alt.Chart(plot)
        .mark_bar(cornerRadiusEnd=5)
        .encode(
            x=alt.X(
                "effect_points:Q",
                title="Relative contribution to 3-year estimate (percentage points)",
            ),
            y=alt.Y("modality:N", sort="-x", title=None),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Raises estimate", "Lowers estimate"],
                    range=[COLORS["coral"], COLORS["teal"]],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            opacity=alt.condition("datum.available", alt.value(1.0), alt.value(0.25)),
            tooltip=[
                "modality:N",
                "direction:N",
                alt.Tooltip("effect_points:Q", format="+.1f"),
            ],
        )
        .properties(height=250)
    )


def _twins_chart(forecast) -> alt.Chart:
    twins = forecast.twin_trajectories.copy()
    twins["series"] = twins["participant_id"] + " · " + twins["cohort"]
    projected = forecast.trajectory[["year", "predicted_score"]].rename(
        columns={"predicted_score": "cognitive_score"}
    )
    projected["series"] = "Selected profile · projection"
    combined = pd.concat(
        [twins[["year", "cognitive_score", "series"]], projected], ignore_index=True
    )
    return (
        alt.Chart(combined)
        .mark_line(point=True)
        .encode(
            x=alt.X("year:Q", title="Years from baseline"),
            y=alt.Y(
                "cognitive_score:Q",
                title="Cognitive composite",
                scale=alt.Scale(domain=[0, 30]),
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    range=[
                        COLORS["teal"],
                        "#7BA69B",
                        "#9BB7AF",
                        "#B8C9C3",
                        "#D5DEDA",
                        COLORS["saffron"],
                    ]
                ),
                legend=alt.Legend(orient="bottom", columns=2, title=None),
            ),
            strokeWidth=alt.condition(
                "datum.series === 'Selected profile · projection'",
                alt.value(4),
                alt.value(1.6),
            ),
            opacity=alt.condition(
                "datum.series === 'Selected profile · projection'",
                alt.value(1),
                alt.value(0.62),
            ),
            tooltip=[
                "series:N",
                alt.Tooltip("year:Q", format=".1f"),
                alt.Tooltip("cognitive_score:Q", format=".1f"),
            ],
        )
        .properties(height=320)
    )


def _participant_view(runtime, forecast) -> None:
    st.markdown("## Participant progression studio")
    safe_id = html.escape(str(forecast.profile.participant_id))
    safe_label = html.escape(str(forecast.profile.label))
    safe_modalities = html.escape(str(len(forecast.available_modalities)))
    st.markdown(
        f"<p class='ns-section-note'>Fictional ID <b>{safe_id}</b> · {safe_label} · "
        f"{safe_modalities} of 5 modality groups available</p>",
        unsafe_allow_html=True,
    )

    risk_columns = st.columns(3)
    # ⚡ Bolt: Replaced slow .iterrows() with .to_dict('records') for UI rendering
    for column, row in zip(risk_columns, forecast.risks.to_dict('records')):
        with column:
            st.markdown(_risk_card(row), unsafe_allow_html=True)

    for warning in forecast.warnings:
        st.warning(warning, icon="⚑")

    left, right = st.columns([1.25, 1])
    with left:
        st.markdown("### Likely cognitive trajectory")
        st.caption(
            "The shaded band combines held-out model residuals with dispersion among matched synthetic participants."
        )
        st.altair_chart(
            _trajectory_chart(forecast.trajectory), use_container_width=True
        )
    with right:
        st.markdown("### What moved the estimate")
        st.caption(
            "Transparent modality-level influence relative to the synthetic training population—not a causal explanation."
        )
        st.altair_chart(_driver_chart(forecast.drivers), use_container_width=True)

    st.markdown("### Digital twin lite · matched trajectories")
    st.caption(
        "Five nearest longitudinal prototypes after train-only imputation and feature standardisation. Similarity is descriptive, not identity matching."
    )
    st.altair_chart(_twins_chart(forecast), use_container_width=True)
    twin_table = forecast.twins.copy()
    twin_table["similarity"] = twin_table["similarity"].map(
        lambda value: f"{value:.0f}%"
    )
    twin_table["baseline_cognition"] = twin_table["baseline_cognition"].map(
        lambda value: f"{value:.1f}"
    )
    st.dataframe(
        twin_table.rename(
            columns={
                "participant_id": "Synthetic participant",
                "cohort": "Reference cohort",
                "urban_rural": "Setting",
                "similarity": "Similarity",
                "baseline_cognition": "Baseline cognition",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )


def _validation_view(runtime) -> None:
    st.markdown("## India-first validation")
    st.markdown(
        "<div class='ns-callout'><b>Evaluation contract:</b> global public cohorts develop the model, TLSA-style data adapt it, and SANSCOG-style rural data remain fully held out. Every value below is calculated from synthetic predictions.</div>",
        unsafe_allow_html=True,
    )
    summary = runtime.validation_summary.copy()
    summary["event_rate"] = summary["event_rate"].map(lambda value: f"{value:.1%}")
    for column in ("auroc", "auprc", "brier"):
        summary[column] = summary[column].map(
            lambda value: "—" if pd.isna(value) else f"{value:.3f}"
        )
    st.dataframe(
        summary.rename(
            columns={
                "validation_set": "Validation set",
                "participants": "Participants",
                "event_rate": "3-year event rate",
                "auroc": "AUROC",
                "auprc": "AUPRC",
                "brier": "Brier",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    calibration_column, ablation_column = st.columns(2)
    with calibration_column:
        st.markdown("### SANSCOG-style calibration")
        calibration = runtime.calibration.copy()
        observed = (
            alt.Chart(calibration)
            .mark_line(point=True, color=COLORS["teal"], strokeWidth=3)
            .encode(
                x=alt.X(
                    "mean_predicted:Q",
                    title="Mean predicted risk",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                y=alt.Y(
                    "observed_rate:Q",
                    title="Observed synthetic rate",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                size=alt.Size("n:Q", legend=None, scale=alt.Scale(range=[40, 160])),
                tooltip=[
                    alt.Tooltip("mean_predicted:Q", format=".2f"),
                    alt.Tooltip("observed_rate:Q", format=".2f"),
                    "n:Q",
                ],
            )
        )
        ideal = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(strokeDash=[5, 5], color="#9AA9A5")
            .encode(x="x:Q", y="y:Q")
        )
        st.altair_chart(
            (ideal + observed).properties(height=310), use_container_width=True
        )

    with ablation_column:
        st.markdown("### Does multimodality help?")
        ablation = runtime.ablation.copy()
        chart = (
            alt.Chart(ablation)
            .mark_bar(cornerRadiusEnd=5, color=COLORS["saffron"])
            .encode(
                x=alt.X("auprc:Q", title="AUPRC", scale=alt.Scale(zero=False)),
                y=alt.Y("scenario:N", sort="-x", title=None),
                tooltip=[
                    "scenario:N",
                    alt.Tooltip("auprc:Q", format=".3f"),
                    alt.Tooltip("auprc_delta:Q", format="+.3f"),
                ],
            )
            .properties(height=310)
        )
        st.altair_chart(chart, use_container_width=True)

    st.markdown("### Indian validation by subgroup")
    st.caption(
        "Small synthetic subgroups are shown transparently; blanks indicate that a group lacked both outcome classes."
    )
    subgroup = runtime.subgroup_metrics.copy()
    chart = (
        alt.Chart(subgroup)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("auprc:Q", title="AUPRC", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("group:N", title=None),
            color=alt.Color(
                "dimension:N",
                scale=alt.Scale(
                    range=[
                        COLORS["teal"],
                        COLORS["saffron"],
                        COLORS["coral"],
                        "#759D8F",
                    ]
                ),
            ),
            column=alt.Column(
                "dimension:N", title=None, header=alt.Header(labelFontSize=13)
            ),
            tooltip=[
                "dimension:N",
                "group:N",
                "participants:Q",
                alt.Tooltip("event_rate:Q", format=".1%"),
                alt.Tooltip("auprc:Q", format=".3f"),
            ],
        )
        .properties(height=230)
    )
    st.altair_chart(chart, use_container_width=True)


def _harmonisation_view(runtime) -> None:
    st.markdown("## Harmonisation and provenance audit")
    st.caption(
        "The demo intentionally begins with cohort-specific aliases and units, then records every mapping into the common longitudinal model."
    )
    metrics = st.columns(4)
    metrics[0].metric(
        "Synthetic participants", f"{len(runtime.bundle.tables.participants):,}"
    )
    metrics[1].metric("Longitudinal visits", f"{len(runtime.bundle.tables.visits):,}")
    metrics[2].metric(
        "Mapped source fields", f"{len(runtime.bundle.harmonization_manifest):,}"
    )
    passed = int((runtime.quality_checks["status"] == "Passed").sum())
    metrics[3].metric("Safety checks passed", f"{passed}/{len(runtime.quality_checks)}")

    st.markdown(
        """
        <div class="ns-flow">
          <div><b>1 · Cohort-native inputs</b>Aliases, units, cadence, and modality gaps</div>
          <div><b>2 · Common data model</b>Participants, visits, features, and outcomes</div>
          <div><b>3 · Train-only transforms</b>Imputation, scaling, and leakage isolation</div>
          <div><b>4 · Auditable outputs</b>Risk, trajectories, twins, and validation</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Cohort-to-common mapping")
    filter_columns = st.columns(2)
    cohort = filter_columns[0].selectbox(
        "Cohort mapping",
        ["All"] + list(runtime.bundle.cohort_summary["cohort"]),
        key="audit_cohort",
    )
    modality = filter_columns[1].selectbox(
        "Modality mapping",
        ["All"] + sorted(runtime.bundle.harmonization_manifest["modality"].unique()),
        key="audit_modality",
    )
    manifest = runtime.bundle.harmonization_manifest.copy()
    if cohort != "All":
        manifest = manifest[manifest["cohort"] == cohort]
    if modality != "All":
        manifest = manifest[manifest["modality"] == modality]
    st.dataframe(manifest, hide_index=True, use_container_width=True)

    missing_column, checks_column = st.columns([1.15, 1])
    with missing_column:
        st.markdown("### Modality completeness")
        heatmap = (
            alt.Chart(runtime.missingness)
            .mark_rect(cornerRadius=3)
            .encode(
                x=alt.X("modality:N", title=None),
                y=alt.Y("cohort:N", title=None),
                color=alt.Color(
                    "missing_rate:Q",
                    title="Missing",
                    scale=alt.Scale(domain=[0, 1], range=["#E3EEE8", COLORS["coral"]]),
                ),
                tooltip=[
                    "cohort:N",
                    "modality:N",
                    alt.Tooltip("missing_rate:Q", format=".0%"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(heatmap, use_container_width=True)
    with checks_column:
        st.markdown("### Leakage and safety checks")
        st.dataframe(
            runtime.quality_checks,
            hide_index=True,
            use_container_width=True,
            height=300,
        )

    st.info(
        "GenomeIndia is used only as the design context for ancestry-aware calibration. ADDI informs the interoperability pattern. YLOPD is excluded from dementia model training.",
        icon="ℹ️",
    )


def main() -> None:
    st.set_page_config(
        page_title="NeuroSaarthi-AD · Progression Studio", page_icon="🧠", layout="wide"
    )
    _inject_styles()
    runtime = _load_runtime()
    profile = _profile_controls()
    forecast = runtime.predict(profile)

    st.markdown(
        """
        <section class="ns-hero">
          <div class="ns-kicker">Healthy brain aging · research prototype</div>
          <h1>NeuroSaarthi-AD</h1>
          <p>A longitudinal multimodal workbench that moves from global evidence to India-specific risk, progression, and matched trajectories—with uncertainty visible at every step.</p>
          <div class="ns-pills">
            <span class="ns-pill">840 synthetic participants</span>
            <span class="ns-pill">7 cohort patterns</span>
            <span class="ns-pill">TLSA adaptation</span>
            <span class="ns-pill">SANSCOG held out</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.error(
        "Synthetic research prototype—not for diagnosis, screening, treatment decisions, or claims about real cohort performance.",
        icon="⚠️",
    )

    participant_tab, validation_tab, harmonisation_tab = st.tabs(
        ["Participant studio", "India-first validation", "Harmonisation audit"]
    )
    with participant_tab:
        _participant_view(runtime, forecast)
    with validation_tab:
        _validation_view(runtime)
    with harmonisation_tab:
        _harmonisation_view(runtime)

    st.divider()
    st.caption(
        "NeuroSaarthi-AD v0.1 · deterministic synthetic data · no uploads, network services, accounts, or participant persistence"
    )


if __name__ == "__main__":
    main()
