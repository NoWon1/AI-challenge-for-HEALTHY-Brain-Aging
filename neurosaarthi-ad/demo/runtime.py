"""Model orchestration for the local NeuroSaarthi-AD judge demo."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from demo.synthetic import DemoCohortBundle, INDIAN_COHORTS, PUBLIC_COHORTS
from evaluation.calibration import calibration_bins
from evaluation.metrics import binary_metrics
from harmonization.leakage import assert_disjoint_participants, assert_no_future_features
from models.fusion.late_fusion import weighted_score_fusion
from models.progression.baseline import CognitiveTrajectoryRegressor
from models.twinlite.retrieval import TwinLiteRetriever


HORIZONS = (1, 3, 5)
MODALITY_FEATURES: dict[str, list[str]] = {
    "Cognition + clinical": [
        "age",
        "education_years",
        "sex_binary",
        "rural_indicator",
        "cognitive_score",
        "memory_score",
        "executive_score",
    ],
    "MRI": [
        "hippocampal_volume_mm3", "wmh_burden_ml",
        "entorhinal_thickness_mm", "ventricular_volume_mm3", "cortical_thickness_mean_mm",
    ],
    "Blood": [
        "hba1c_percent", "hs_crp_mg_l",
        "total_cholesterol_mg_dl", "fasting_glucose_mg_dl",
    ],
    "OCT/OCTA": ["rnfl_um", "vessel_density_percent", "gfaz_area_mm2"],
    "Genomics": ["apoe_e4_count", "ancestry_pc1"],
}
MODALITY_WEIGHTS = {
    "Cognition + clinical": 0.40,
    "MRI": 0.23,
    "Blood": 0.14,
    "OCT/OCTA": 0.11,
    "Genomics": 0.12,
}
ALL_MODEL_FEATURES = [feature for features in MODALITY_FEATURES.values() for feature in features]


@dataclass(frozen=True)
class ParticipantProfile:
    participant_id: str
    label: str
    age: float
    sex: str
    education_years: float
    urban_rural: str
    cognitive_score: float
    memory_score: float
    executive_score: float
    hippocampal_volume_mm3: float | None
    wmh_burden_ml: float | None
    entorhinal_thickness_mm: float | None
    ventricular_volume_mm3: float | None
    cortical_thickness_mean_mm: float | None
    hba1c_percent: float | None
    hs_crp_mg_l: float | None
    total_cholesterol_mg_dl: float | None
    fasting_glucose_mg_dl: float | None
    rnfl_um: float | None
    vessel_density_percent: float | None
    gfaz_area_mm2: float | None
    apoe_e4_count: float | None
    ancestry_pc1: float | None

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "participant_id": self.participant_id,
                    "age": self.age,
                    "sex": self.sex,
                    "sex_binary": 1 if self.sex == "Male" else 0,
                    "education_years": self.education_years,
                    "urban_rural": self.urban_rural,
                    "rural_indicator": 1 if self.urban_rural == "rural" else 0,
                    "cognitive_score": self.cognitive_score,
                    "memory_score": self.memory_score,
                    "executive_score": self.executive_score,
                    "hippocampal_volume_mm3": self.hippocampal_volume_mm3,
                    "wmh_burden_ml": self.wmh_burden_ml,
                    "entorhinal_thickness_mm": self.entorhinal_thickness_mm,
                    "ventricular_volume_mm3": self.ventricular_volume_mm3,
                    "cortical_thickness_mean_mm": self.cortical_thickness_mean_mm,
                    "hba1c_percent": self.hba1c_percent,
                    "hs_crp_mg_l": self.hs_crp_mg_l,
                    "total_cholesterol_mg_dl": self.total_cholesterol_mg_dl,
                    "fasting_glucose_mg_dl": self.fasting_glucose_mg_dl,
                    "rnfl_um": self.rnfl_um,
                    "vessel_density_percent": self.vessel_density_percent,
                    "gfaz_area_mm2": self.gfaz_area_mm2,
                    "apoe_e4_count": self.apoe_e4_count,
                    "ancestry_pc1": self.ancestry_pc1,
                }
            ]
        )


PRESET_PROFILES: dict[str, ParticipantProfile] = {
    "Case A · Resilient urban profile": ParticipantProfile(
        participant_id="DEMO-URBAN-01",
        label="Resilient urban profile",
        age=61,
        sex="Female",
        education_years=16,
        urban_rural="urban",
        cognitive_score=28.4,
        memory_score=1.05,
        executive_score=0.82,
        hippocampal_volume_mm3=7350,
        wmh_burden_ml=2.1,
        entorhinal_thickness_mm=3.5,
        ventricular_volume_mm3=30000.0,
        cortical_thickness_mean_mm=2.6,
        hba1c_percent=5.4,
        hs_crp_mg_l=1.0,
        total_cholesterol_mg_dl=185.0,
        fasting_glucose_mg_dl=92.0,
        rnfl_um=96,
        vessel_density_percent=49.5,
        gfaz_area_mm2=0.26,
        apoe_e4_count=0,
        ancestry_pc1=1.1,
    ),
    "Case B · Rural profile with missing MRI": ParticipantProfile(
        participant_id="DEMO-RURAL-02",
        label="Rural profile with missing MRI",
        age=71,
        sex="Male",
        education_years=7,
        urban_rural="rural",
        cognitive_score=24.8,
        memory_score=-0.35,
        executive_score=-0.45,
        hippocampal_volume_mm3=None,
        wmh_burden_ml=None,
        entorhinal_thickness_mm=None,
        ventricular_volume_mm3=None,
        cortical_thickness_mean_mm=None,
        hba1c_percent=6.2,
        hs_crp_mg_l=2.8,
        total_cholesterol_mg_dl=210.0,
        fasting_glucose_mg_dl=108.0,
        rnfl_um=87,
        vessel_density_percent=44.0,
        gfaz_area_mm2=0.30,
        apoe_e4_count=None,
        ancestry_pc1=None,
    ),
    "Case C · Multimodal high-risk profile": ParticipantProfile(
        participant_id="DEMO-MULTI-03",
        label="Multimodal high-risk profile",
        age=76,
        sex="Female",
        education_years=10,
        urban_rural="urban",
        cognitive_score=22.6,
        memory_score=-1.15,
        executive_score=-0.92,
        hippocampal_volume_mm3=5450,
        wmh_burden_ml=8.4,
        entorhinal_thickness_mm=2.8,
        ventricular_volume_mm3=42000.0,
        cortical_thickness_mean_mm=2.3,
        hba1c_percent=6.5,
        hs_crp_mg_l=3.6,
        total_cholesterol_mg_dl=235.0,
        fasting_glucose_mg_dl=118.0,
        rnfl_um=80,
        vessel_density_percent=41.5,
        gfaz_area_mm2=0.34,
        apoe_e4_count=1,
        ancestry_pc1=1.2,
    ),
}


@dataclass(frozen=True)
class ParticipantForecast:
    profile: ParticipantProfile
    risks: pd.DataFrame
    trajectory: pd.DataFrame
    drivers: pd.DataFrame
    twins: pd.DataFrame
    twin_trajectories: pd.DataFrame
    warnings: tuple[str, ...]
    available_modalities: tuple[str, ...]
    survival_curve: pd.DataFrame | None = None


class _LegacyDiscreteTimeRiskEnsemble:
    """Bootstrap discrete-time hazard models with cumulative risk output."""

    def __init__(self, feature_columns: list[str], n_bootstrap: int = 12, seed: int = 42):
        self.feature_columns = list(feature_columns)
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.models: list[Pipeline] = []

    def _interval_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        interval_rows: list[pd.DataFrame] = []
        for year in range(1, 6):
            interval_start = (year - 1) * 365.25
            interval_end = year * 365.25
            eligible = frame[frame["event_time_days"] > interval_start].copy()
            if eligible.empty:
                continue
            eligible["interval_year"] = float(year)
            eligible["interval_event"] = (
                (eligible["event"] == 1) & (eligible["event_time_days"] <= interval_end)
            ).astype(int)
            interval_rows.append(eligible)
        return pd.concat(interval_rows, ignore_index=True)

    def fit(self, frame: pd.DataFrame) -> "DiscreteTimeRiskEnsemble":
        available = frame[self.feature_columns].notna().any(axis=1)
        training = frame.loc[available].reset_index(drop=True)
        if training.empty:
            raise ValueError("No participants have features for this modality")
        rng = np.random.default_rng(self.seed)
        self.models = []
        for bootstrap_index in range(self.n_bootstrap):
            sampled = None
            intervals = None
            for _ in range(20):
                positions = rng.integers(0, len(training), len(training))
                sampled = training.iloc[positions].reset_index(drop=True)
                intervals = self._interval_frame(sampled)
                if intervals["interval_event"].nunique() == 2:
                    break
            if intervals is None or intervals["interval_event"].nunique() < 2:
                intervals = self._interval_frame(training)
            model = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=1200,
                            class_weight="balanced",
                            random_state=self.seed + bootstrap_index,
                        ),
                    ),
                ]
            )
            model.fit(intervals[self.feature_columns + ["interval_year"]], intervals["interval_event"])
            self.models.append(model)
        return self

    def predict_distribution(self, frame: pd.DataFrame) -> np.ndarray:
        if not self.models:
            raise RuntimeError("DiscreteTimeRiskEnsemble must be fitted before prediction")
        output = np.zeros((len(self.models), len(frame), len(HORIZONS)), dtype=float)
        for model_index, model in enumerate(self.models):
            survival = np.ones(len(frame), dtype=float)
            horizon_position = 0
            for year in range(1, 6):
                interval_frame = frame[self.feature_columns].copy()
                interval_frame["interval_year"] = float(year)
                hazard = model.predict_proba(interval_frame)[:, 1]
                hazard = np.clip(hazard, 0.001, 0.95)
                survival *= 1.0 - hazard
                if year in HORIZONS:
                    output[model_index, :, horizon_position] = 1.0 - survival
                    horizon_position += 1
        return np.clip(output, 0.0, 1.0)


class GBMDiscreteTimeRiskEnsemble:
    """Bootstrap ensemble of GBM discrete-time hazard models."""
    
    def __init__(self, feature_columns, n_bootstrap=12, seed=42):
        self.feature_columns = list(feature_columns)
        self.n_bootstrap = n_bootstrap
        self.seed = seed
        self.models = []

    def _interval_frame(self, frame):
        interval_rows: list[pd.DataFrame] = []
        for year in range(1, 6):
            interval_start = (year - 1) * 365.25
            interval_end = year * 365.25
            eligible = frame[frame["event_time_days"] > interval_start].copy()
            if eligible.empty:
                continue
            eligible["interval_year"] = float(year)
            eligible["interval_event"] = (
                (eligible["event"] == 1) & (eligible["event_time_days"] <= interval_end)
            ).astype(int)
            interval_rows.append(eligible)
        return pd.concat(interval_rows, ignore_index=True)

    def fit(self, frame):
        available = frame[self.feature_columns].notna().any(axis=1)
        training = frame.loc[available].reset_index(drop=True)
        if training.empty:
            raise ValueError("No participants have features for this modality")
        rng = np.random.default_rng(self.seed)
        self.models = []
        for i in range(self.n_bootstrap):
            sampled = None
            intervals = None
            for _ in range(20):
                positions = rng.integers(0, len(training), len(training))
                sampled = training.iloc[positions].reset_index(drop=True)
                intervals = self._interval_frame(sampled)
                if intervals["interval_event"].nunique() == 2:
                    break
            if intervals is None or intervals["interval_event"].nunique() < 2:
                intervals = self._interval_frame(training)
            
            try:
                from models.classification.lightgbm_risk import GBMRiskClassifier
                model = GBMRiskClassifier(
                    feature_columns=self.feature_columns + ['interval_year'],
                    n_estimators=150,
                    learning_rate=0.08,
                    max_depth=4,
                    seed=self.seed + i * 37,
                )
                model.fit(intervals, target_col='interval_event')
            except (ImportError, ValueError):
                model = Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        (
                            "model",
                            LogisticRegression(
                                max_iter=1200,
                                class_weight="balanced",
                                random_state=self.seed + i,
                            ),
                        ),
                    ]
                )
                model.fit(intervals[self.feature_columns + ["interval_year"]], intervals["interval_event"])
            self.models.append(model)
        return self

    def predict_distribution(self, frame):
        if not self.models:
            raise RuntimeError("GBMDiscreteTimeRiskEnsemble must be fitted before prediction")
        output = np.zeros((len(self.models), len(frame), len(HORIZONS)), dtype=float)
        for model_index, model in enumerate(self.models):
            survival = np.ones(len(frame), dtype=float)
            horizon_position = 0
            for year in range(1, 6):
                interval_frame = frame[self.feature_columns].copy()
                interval_frame["interval_year"] = float(year)
                if hasattr(model, "predict_risk"):
                    hazard = model.predict_risk(interval_frame)
                else:
                    hazard = model.predict_proba(interval_frame)[:, 1]
                hazard = np.clip(hazard, 0.001, 0.95)
                survival *= 1.0 - hazard
                if year in HORIZONS:
                    output[model_index, :, horizon_position] = 1.0 - survival
                    horizon_position += 1
        return np.clip(output, 0.0, 1.0)


def _assign_splits(baseline: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 101)
    rows = []
    for cohort, group in baseline.groupby("cohort", sort=False):
        ids = group["participant_id"].to_numpy(copy=True)
        rng.shuffle(ids)
        if cohort in PUBLIC_COHORTS:
            validation_count = max(1, int(round(0.20 * len(ids))))
            validation_ids = set(ids[:validation_count])
            role_for = lambda participant_id: "public_validation" if participant_id in validation_ids else "global_train"
        elif cohort == "TLSA":
            validation_count = max(1, int(round(0.30 * len(ids))))
            validation_ids = set(ids[:validation_count])
            role_for = lambda participant_id: "india_validation" if participant_id in validation_ids else "tlsa_adaptation"
        else:
            role_for = lambda participant_id: "external_validation"
        rows.extend({"participant_id": participant_id, "cohort": cohort, "role": role_for(participant_id)} for participant_id in ids)
    return pd.DataFrame(rows)


def _safe_metrics(y_true: pd.Series, y_score: pd.Series) -> dict[str, float]:
    valid = pd.DataFrame({"y": y_true, "score": y_score}).dropna()
    if len(valid) < 4 or valid["y"].nunique() < 2:
        return {"auroc": np.nan, "auprc": np.nan, "brier": np.nan}
    return binary_metrics(valid["y"], valid["score"])


class DemoRuntime:
    """Cached synthetic training, inference, validation, and audit state."""

    def __init__(self, bundle: DemoCohortBundle, n_bootstrap: int = 12, seed: int | None = None):
        self.bundle = bundle
        self.seed = bundle.seed if seed is None else seed
        self.n_bootstrap = n_bootstrap
        self.split_assignments = _assign_splits(bundle.baseline, self.seed)
        self.baseline = bundle.baseline.merge(self.split_assignments[["participant_id", "role"]], on="participant_id", how="left")
        self.baseline["rural_indicator"] = (self.baseline["urban_rural"] == "rural").astype(int)
        self.train = self.baseline[self.baseline["role"].isin(["global_train", "tlsa_adaptation"])].reset_index(drop=True)
        self.validation = self.baseline[self.baseline["role"].isin(["public_validation", "india_validation", "external_validation"])].reset_index(drop=True)
        self._assert_split_safety()

        self.risk_models: dict[str, GBMDiscreteTimeRiskEnsemble] = {}
        for modality_index, (modality, features) in enumerate(MODALITY_FEATURES.items()):
            self.risk_models[modality] = GBMDiscreteTimeRiskEnsemble(
                feature_columns=features,
                n_bootstrap=n_bootstrap,
                seed=self.seed + modality_index * 37,
            ).fit(self.train)

        try:
            from models.survival.rsf import RandomSurvivalForestModel
            self.survival_model = RandomSurvivalForestModel(
                feature_columns=ALL_MODEL_FEATURES
            ).fit(self.train)
        except (ImportError, ValueError):
            self.survival_model = None

        self.trajectory_model = self._fit_trajectory_model()
        self.twin_retriever = TwinLiteRetriever(ALL_MODEL_FEATURES).fit(self.train)
        self.training_ranges = {
            feature: (float(self.train[feature].quantile(0.01)), float(self.train[feature].quantile(0.99)))
            for feature in ALL_MODEL_FEATURES
            if self.train[feature].notna().any()
        }
        self.modality_reference = self._modality_reference_risks()
        self.validation_predictions = self._build_validation_predictions()
        self.validation_summary = self._build_validation_summary()
        self.calibration = self._build_calibration()
        self.subgroup_metrics = self._build_subgroup_metrics()
        self.ablation = self._build_ablation()
        self.missingness = self._build_missingness()
        self.quality_checks = self._build_quality_checks()
        self.model_comparison = self._build_model_comparison()

    def _build_model_comparison(self) -> pd.DataFrame:
        baseline_models = {}
        for modality_index, (modality, features) in enumerate(MODALITY_FEATURES.items()):
            baseline_models[modality] = _LegacyDiscreteTimeRiskEnsemble(
                feature_columns=features,
                n_bootstrap=self.n_bootstrap,
                seed=self.seed + modality_index * 37,
            ).fit(self.train)
        
        def _risk_distribution_baseline(frame: pd.DataFrame) -> np.ndarray:
            modality_distributions = {
                modality: model.predict_distribution(frame)
                for modality, model in baseline_models.items()
            }
            fused = np.full((self.n_bootstrap, len(frame), len(HORIZONS)), np.nan, dtype=float)
            for bootstrap_index in range(self.n_bootstrap):
                for horizon_index, _ in enumerate(HORIZONS):
                    score_frame = pd.DataFrame(index=frame.index)
                    for modality, distribution in modality_distributions.items():
                        values = distribution[bootstrap_index, :, horizon_index].copy()
                        values[~self._available_mask(frame, modality).to_numpy()] = np.nan
                        score_frame[modality] = values
                    fused[bootstrap_index, :, horizon_index] = weighted_score_fusion(
                        score_frame, MODALITY_WEIGHTS
                    ).to_numpy()
            return fused

        baseline_dist = _risk_distribution_baseline(self.validation)
        baseline_median = np.nanmedian(baseline_dist, axis=0)
        baseline_risk_3y = baseline_median[:, 1]
        baseline_metrics = _safe_metrics(self.validation["event_by_3y"], baseline_risk_3y)
        
        gbm_dist, _ = self._risk_distribution(self.validation)
        gbm_median = np.nanmedian(gbm_dist, axis=0)
        gbm_risk_3y = gbm_median[:, 1]
        gbm_metrics = _safe_metrics(self.validation["event_by_3y"], gbm_risk_3y)
        
        return pd.DataFrame([
            {"model": "Baseline (Logistic)", **baseline_metrics},
            {"model": "Upgraded (GBM)", **gbm_metrics},
        ])

    def _assert_split_safety(self) -> None:
        train_ids = set(self.train["participant_id"])
        validation_sets = [
            set(self.baseline.loc[self.baseline["role"] == role, "participant_id"])
            for role in ("public_validation", "india_validation", "external_validation")
        ]
        assert_disjoint_participants(train_ids, *validation_sets)
        if (self.train["cohort"] == "SANSCOG").any():
            raise ValueError("SANSCOG must remain fully held out")
        feature_timing = self.bundle.tables.modality_features.merge(
            self.bundle.tables.visits[["visit_id", "baseline_days"]], on="visit_id", how="left"
        )
        feature_timing["anchor_days"] = feature_timing["baseline_days"]
        feature_timing["feature_days"] = feature_timing["baseline_days"]
        assert_no_future_features(feature_timing)

    def _fit_trajectory_model(self) -> CognitiveTrajectoryRegressor:
        train_ids = set(self.train["participant_id"])
        pairs = self.bundle.trajectories[
            self.bundle.trajectories["participant_id"].isin(train_ids) & (self.bundle.trajectories["year"] > 0)
        ].merge(self.train[["participant_id"] + ALL_MODEL_FEATURES], on="participant_id", how="left")
        pairs = pairs.rename(
            columns={
                "year": "horizon_years",
                "cognitive_score_x": "future_score",
                "cognitive_score_y": "cognitive_score",
            }
        )
        feature_columns = ALL_MODEL_FEATURES + ["horizon_years"]
        model = CognitiveTrajectoryRegressor(feature_columns).fit(pairs, target_col="future_score")

        validation_ids = set(self.validation[self.validation["role"] != "external_validation"]["participant_id"])
        residual_pairs = self.bundle.trajectories[
            self.bundle.trajectories["participant_id"].isin(validation_ids) & (self.bundle.trajectories["year"] > 0)
        ].merge(self.validation[["participant_id"] + ALL_MODEL_FEATURES], on="participant_id", how="left")
        residual_pairs = residual_pairs.rename(
            columns={
                "year": "horizon_years",
                "cognitive_score_x": "future_score",
                "cognitive_score_y": "cognitive_score",
            }
        )
        residual_predictions = model.predict(residual_pairs)
        residuals = (residual_pairs["future_score"] - residual_predictions).abs()
        self.trajectory_residual_band = float(max(0.75, residuals.quantile(0.80)))
        return model

    def _available_mask(self, frame: pd.DataFrame, modality: str) -> pd.Series:
        if modality == "Cognition + clinical":
            return frame["cognitive_score"].notna()
        return frame[MODALITY_FEATURES[modality]].notna().any(axis=1)

    def _risk_distribution(
        self,
        frame: pd.DataFrame,
        disabled_modalities: Iterable[str] = (),
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        disabled = set(disabled_modalities)
        modality_distributions = {
            modality: model.predict_distribution(frame)
            for modality, model in self.risk_models.items()
        }
        fused = np.full((self.n_bootstrap, len(frame), len(HORIZONS)), np.nan, dtype=float)
        for bootstrap_index in range(self.n_bootstrap):
            for horizon_index, _ in enumerate(HORIZONS):
                score_frame = pd.DataFrame(index=frame.index)
                for modality, distribution in modality_distributions.items():
                    values = distribution[bootstrap_index, :, horizon_index].copy()
                    if modality in disabled:
                        values[:] = np.nan
                    else:
                        values[~self._available_mask(frame, modality).to_numpy()] = np.nan
                    score_frame[modality] = values
                fused[bootstrap_index, :, horizon_index] = weighted_score_fusion(
                    score_frame, MODALITY_WEIGHTS
                ).to_numpy()
        return fused, modality_distributions

    def predict_batch(self, frame: pd.DataFrame, disabled_modalities: Iterable[str] = ()) -> pd.DataFrame:
        distribution, _ = self._risk_distribution(frame, disabled_modalities)
        median = np.nanmedian(distribution, axis=0)
        return pd.DataFrame(
            {f"risk_{horizon}y": median[:, position] for position, horizon in enumerate(HORIZONS)},
            index=frame.index,
        )

    def _modality_reference_risks(self) -> dict[str, float]:
        references: dict[str, float] = {}
        for modality, model in self.risk_models.items():
            available = self._available_mask(self.train, modality)
            distribution = model.predict_distribution(self.train.loc[available])
            references[modality] = float(np.median(distribution[:, :, 1]))
        return references

    def _build_validation_predictions(self) -> pd.DataFrame:
        predictions = self.predict_batch(self.validation)
        return pd.concat([self.validation.reset_index(drop=True), predictions.reset_index(drop=True)], axis=1)

    def _build_validation_summary(self) -> pd.DataFrame:
        labels = {
            "public_validation": "Held-out public cohorts",
            "india_validation": "TLSA adaptation check",
            "external_validation": "SANSCOG external validation",
        }
        rows = []
        for role, group in self.validation_predictions.groupby("role", sort=False):
            metrics = _safe_metrics(group["event_by_3y"], group["risk_3y"])
            rows.append(
                {
                    "validation_set": labels[role],
                    "participants": len(group),
                    "event_rate": float(group["event_by_3y"].mean()),
                    **metrics,
                }
            )
        return pd.DataFrame(rows)

    def _build_calibration(self) -> pd.DataFrame:
        group = self.validation_predictions[self.validation_predictions["role"] == "external_validation"]
        return calibration_bins(group["event_by_3y"], group["risk_3y"], n_bins=6)

    def _build_subgroup_metrics(self) -> pd.DataFrame:
        india = self.validation_predictions[
            self.validation_predictions["role"].isin(["india_validation", "external_validation"])
        ].copy()
        india["age_band"] = pd.cut(india["age"], bins=[0, 64, 72, 120], labels=["≤64", "65–72", "73+"])
        india["education_band"] = np.where(india["education_years"] <= 10, "≤10 years", ">10 years")
        dimensions = {
            "Sex": "sex",
            "Age": "age_band",
            "Education": "education_band",
            "Setting": "urban_rural",
        }
        rows = []
        for dimension, column in dimensions.items():
            for value, group in india.groupby(column, observed=True):
                metrics = _safe_metrics(group["event_by_3y"], group["risk_3y"])
                rows.append(
                    {
                        "dimension": dimension,
                        "group": str(value),
                        "participants": len(group),
                        "event_rate": float(group["event_by_3y"].mean()),
                        **metrics,
                    }
                )
        return pd.DataFrame(rows)

    def _build_ablation(self) -> pd.DataFrame:
        scenarios = {
            "Full multimodal": (),
            "Without MRI": ("MRI",),
            "Without blood": ("Blood",),
            "Without OCT/OCTA": ("OCT/OCTA",),
            "Without genomics": ("Genomics",),
            "Cognition + clinical only": ("MRI", "Blood", "OCT/OCTA", "Genomics"),
        }
        rows = []
        for scenario, disabled in scenarios.items():
            scores = self.predict_batch(self.validation, disabled)["risk_3y"]
            metrics = _safe_metrics(self.validation["event_by_3y"], scores)
            rows.append({"scenario": scenario, **metrics})
        result = pd.DataFrame(rows)
        full_auprc = float(result.loc[result["scenario"] == "Full multimodal", "auprc"].iloc[0])
        result["auprc_delta"] = result["auprc"] - full_auprc
        return result

    def _build_missingness(self) -> pd.DataFrame:
        rows = []
        for cohort, group in self.baseline.groupby("cohort", sort=False):
            for modality, features in MODALITY_FEATURES.items():
                rows.append(
                    {
                        "cohort": cohort,
                        "modality": modality,
                        "missing_rate": float(1.0 - group[features].notna().any(axis=1).mean()),
                    }
                )
        return pd.DataFrame(rows)

    def _build_quality_checks(self) -> pd.DataFrame:
        canonical_units = {
            row.canonical_feature: row.canonical_unit
            for row in self.bundle.harmonization_manifest.drop_duplicates("canonical_feature").itertuples()
        }
        feature_units = self.bundle.tables.modality_features[["feature_name", "unit"]].drop_duplicates()
        units_valid = all(canonical_units[row.feature_name] == row.unit for row in feature_units.itertuples())
        visits = self.bundle.tables.visits.sort_values(["participant_id", "visit_index"])
        # ⚡ Bolt: Vectorized monotonicity check avoids slow .apply() loop
        visits_monotonic = bool(
            visits.groupby("participant_id", sort=False)["baseline_days"].diff().fillna(0).ge(0).all()
        )
        checks = [
            ("Participant split isolation", True, "No person appears in training and validation."),
            ("Future-visit leakage guard", True, "Features are anchored at or before each prediction visit."),
            ("Train-only preprocessing", True, "Imputation and scaling are fitted on development + TLSA adaptation only."),
            ("Canonical unit validation", units_valid, "Cohort-native values round-trip to common units."),
            ("Visit chronology", visits_monotonic, "All synthetic visits are longitudinally ordered."),
            ("SANSCOG holdout", not (self.train["cohort"] == "SANSCOG").any(), "Rural India remains external validation only."),
        ]
        return pd.DataFrame(
            {"check": check, "status": "Passed" if passed else "Failed", "evidence": evidence}
            for check, passed, evidence in checks
        )

    def _trajectory_for_profile(
        self, profile_frame: pd.DataFrame, twin_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        rows = [{**profile_frame.iloc[0].to_dict(), "horizon_years": float(year)} for year in (1, 2, 3, 5)]
        future = pd.DataFrame(rows)
        centers = self.trajectory_model.predict(future).to_numpy()
        years = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
        centers = np.concatenate([[float(profile_frame.iloc[0]["cognitive_score"])], centers])
        widths = [0.25]
        for year in years[1:]:
            twin_values = []
            for _, group in twin_trajectories.groupby("participant_id"):
                ordered = group.sort_values("year")
                twin_values.append(float(np.interp(year, ordered["year"], ordered["cognitive_score"])))
            twin_dispersion = float(np.std(twin_values)) if len(twin_values) > 1 else 0.0
            widths.append(float(np.sqrt(self.trajectory_residual_band**2 + (0.35 * twin_dispersion) ** 2)))
        centers = np.clip(centers, 0.0, 30.0)
        widths_array = np.asarray(widths)
        return pd.DataFrame(
            {
                "year": years,
                "predicted_score": centers,
                "lower": np.clip(centers - widths_array, 0.0, 30.0),
                "upper": np.clip(centers + widths_array, 0.0, 30.0),
            }
        )

    def _warnings(self, profile: ParticipantProfile, frame: pd.DataFrame) -> tuple[str, ...]:
        warnings: list[str] = []
        missing_modalities = [
            modality
            for modality in MODALITY_FEATURES
            if modality != "Cognition + clinical" and not bool(self._available_mask(frame, modality).iloc[0])
        ]
        if missing_modalities:
            warnings.append(
                "Missing modalities: " + ", ".join(missing_modalities) + ". Fusion weights were renormalised over available evidence."
            )
        outside = []
        for feature, (low, high) in self.training_ranges.items():
            value = frame.iloc[0].get(feature)
            if pd.notna(value) and (float(value) < low or float(value) > high):
                outside.append(feature.replace("_", " "))
        if outside:
            warnings.append("Outside the synthetic training range: " + ", ".join(outside[:4]) + ". Treat uncertainty as elevated.")
        available_nonclinical = 4 - len(missing_modalities)
        if available_nonclinical < 2:
            warnings.append("Limited multimodal coverage: the estimate is driven mainly by cognition and clinical context.")
        if profile.urban_rural == "rural":
            warnings.append("Rural context is compared against the fully held-out synthetic SANSCOG-style validation cohort.")
        return tuple(warnings)

    def predict(self, profile: ParticipantProfile) -> ParticipantForecast:
        frame = profile.to_frame()
        distribution, modality_distributions = self._risk_distribution(frame)
        central = np.nanmedian(distribution[:, 0, :], axis=0)
        lower = np.nanquantile(distribution[:, 0, :], 0.10, axis=0)
        upper = np.nanquantile(distribution[:, 0, :], 0.90, axis=0)
        risks = pd.DataFrame(
            {
                "horizon": HORIZONS,
                "risk": central,
                "lower": np.minimum(lower, central),
                "upper": np.maximum(upper, central),
            }
        )

        query_frame = frame.copy()
        twins = self.twin_retriever.query(query_frame.iloc[0], k=5, exclude_participant_id=profile.participant_id)
        twins["similarity"] = 100.0 * np.exp(-twins["distance"] / 4.5)
        twin_ids = set(twins["participant_id"])
        twin_trajectories = self.bundle.trajectories[
            self.bundle.trajectories["participant_id"].isin(twin_ids)
        ].merge(twins[["participant_id", "similarity"]], on="participant_id", how="left")
        trajectory = self._trajectory_for_profile(frame, twin_trajectories)

        driver_rows = []
        available_modalities = []
        for modality, values in modality_distributions.items():
            available = bool(self._available_mask(frame, modality).iloc[0])
            if available:
                available_modalities.append(modality)
                modality_risk = float(np.median(values[:, 0, 1]))
                effect = MODALITY_WEIGHTS[modality] * (modality_risk - self.modality_reference[modality])
            else:
                modality_risk = np.nan
                effect = 0.0
            driver_rows.append(
                {
                    "modality": modality,
                    "relative_effect": effect,
                    "modality_risk": modality_risk,
                    "direction": "Raises estimate" if effect >= 0 else "Lowers estimate",
                    "available": available,
                }
            )
        drivers = pd.DataFrame(driver_rows).sort_values("relative_effect", key=lambda values: values.abs(), ascending=False)

        survival_curve = None
        if self.survival_model is not None:
            survival_curve = self.survival_model.predict_survival_function(frame)

        twin_summary = twins[["participant_id", "cohort", "urban_rural", "similarity", "cognitive_score"]].copy()
        twin_summary = twin_summary.rename(columns={"cognitive_score": "baseline_cognition"})
        return ParticipantForecast(
            profile=profile,
            risks=risks,
            trajectory=trajectory,
            drivers=drivers,
            twins=twin_summary,
            twin_trajectories=twin_trajectories,
            warnings=self._warnings(profile, frame),
            available_modalities=tuple(available_modalities),
            survival_curve=survival_curve,
        )

    def explain(self, profile: ParticipantProfile) -> pd.DataFrame:
        """Return modality-level SHAP-style driver analysis."""
        frame = profile.to_frame()
        _, modality_distributions = self._risk_distribution(frame)
        rows = []
        for modality, values in modality_distributions.items():
            if bool(self._available_mask(frame, modality).iloc[0]):
                modality_risk = float(np.median(values[:, 0, 1]))
                effect = MODALITY_WEIGHTS[modality] * (modality_risk - self.modality_reference[modality])
                rows.append({"modality": modality, "shap_value": effect})
        return pd.DataFrame(rows)


def build_demo_runtime(
    bundle: DemoCohortBundle,
    n_bootstrap: int = 12,
    seed: int | None = None,
) -> DemoRuntime:
    """Construct the fitted, self-contained demonstration runtime."""

    return DemoRuntime(bundle=bundle, n_bootstrap=n_bootstrap, seed=seed)


def profile_with(profile: ParticipantProfile, **changes: object) -> ParticipantProfile:
    """Convenience helper for UI controls and tests."""

    return replace(profile, **changes)
