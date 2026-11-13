import re

with open("neurosaarthi-ad/demo/runtime.py", "r") as f:
    content = f.read()

# Fix 1: duplicate fit method
content = re.sub(
    r"    def fit\(self, frame: pd\.DataFrame\) -> DiscreteTimeRiskEnsemble:\n    def fit\(self, frame: pd\.DataFrame\) -> _LegacyDiscreteTimeRiskEnsemble:",
    r"    def fit(self, frame: pd.DataFrame) -> _LegacyDiscreteTimeRiskEnsemble:",
    content
)

content = content.replace(
"""        visits_monotonic = bool(
            visits.groupby("participant_id")["baseline_days"]
            .is_monotonic_increasing.all()
        # ⚡ Bolt: Vectorized monotonicity check avoids slow .apply() loop and slow .groupby() overhead
        pids = visits["participant_id"].to_numpy()
        days = visits["baseline_days"].to_numpy()
        same_pid = pids[1:] == pids[:-1]
        diffs = days[1:] - days[:-1]
        visits_monotonic = (
            bool(np.all(diffs[same_pid] >= 0)) if len(visits) > 1 else True
        )""",
"""        # ⚡ Bolt: Vectorized monotonicity check avoids slow .apply() loop and slow .groupby() overhead
        pids = visits["participant_id"].to_numpy()
        days = visits["baseline_days"].to_numpy()
        same_pid = pids[1:] == pids[:-1]
        diffs = days[1:] - days[:-1]
        visits_monotonic = (
            bool(np.all(diffs[same_pid] >= 0)) if len(visits) > 1 else True
        )""")

content = content.replace(
"""        visits_monotonic = bool(
            visits.groupby("participant_id", sort=False)["baseline_days"]
            .diff()
            .fillna(0)
            .ge(0)
            .all()
        # ⚡ Bolt: Vectorized monotonicity check avoids slow .apply() loop and slow .groupby() overhead
        pids = visits["participant_id"].to_numpy()
        days = visits["baseline_days"].to_numpy()
        same_pid = pids[1:] == pids[:-1]
        diffs = days[1:] - days[:-1]
        visits_monotonic = (
            bool(np.all(diffs[same_pid] >= 0)) if len(visits) > 1 else True
        )""",
"""        # ⚡ Bolt: Vectorized monotonicity check avoids slow .apply() loop and slow .groupby() overhead
        pids = visits["participant_id"].to_numpy()
        days = visits["baseline_days"].to_numpy()
        same_pid = pids[1:] == pids[:-1]
        diffs = days[1:] - days[:-1]
        visits_monotonic = (
            bool(np.all(diffs[same_pid] >= 0)) if len(visits) > 1 else True
        )""")


with open("neurosaarthi-ad/demo/runtime.py", "w") as f:
    f.write(content)

with open("neurosaarthi-ad/models/progression/baseline.py", "r") as f:
    content = f.read()
content = content.replace("def fit(\n        self, frame: pd.DataFrame, target_col: str = \"future_score\"\n    ) -> CognitiveTrajectoryRegressor:\n    def fit(self, frame: pd.DataFrame, target_col: str = \"future_score\") -> CognitiveTrajectoryRegressor:", "def fit(self, frame: pd.DataFrame, target_col: str = \"future_score\") -> CognitiveTrajectoryRegressor:")
with open("neurosaarthi-ad/models/progression/baseline.py", "w") as f:
    f.write(content)

with open("neurosaarthi-ad/models/classification/lightgbm_risk.py", "r") as f:
    content = f.read()
content = content.replace("def fit(self, frame: pd.DataFrame, target_col: str = \"event\") -> GBMRiskClassifier:\n    def fit(\n        self, frame: pd.DataFrame, target_col: str = \"event\"\n    ) -> GBMRiskClassifier:", "def fit(self, frame: pd.DataFrame, target_col: str = \"event\") -> GBMRiskClassifier:")
with open("neurosaarthi-ad/models/classification/lightgbm_risk.py", "w") as f:
    f.write(content)

with open("neurosaarthi-ad/models/survival/gradient_boosting.py", "r") as f:
    content = f.read()
content = content.replace("def fit(\n        self,\n        frame: pd.DataFrame,\n        time_col: str = \"event_time_days\",\n        event_col: str = \"event\",\n    ) -> GradientBoostingSurvivalModel:\n    def fit(self, frame: pd.DataFrame, time_col: str = \"event_time_days\", event_col: str = \"event\") -> GradientBoostingSurvivalModel:", "def fit(self, frame: pd.DataFrame, time_col: str = \"event_time_days\", event_col: str = \"event\") -> GradientBoostingSurvivalModel:")
with open("neurosaarthi-ad/models/survival/gradient_boosting.py", "w") as f:
    f.write(content)

with open("neurosaarthi-ad/models/fusion/stacking.py", "r") as f:
    content = f.read()
content = content.replace("def fit(\n        self,\n        base_predictions_dict: dict[str, np.ndarray | pd.Series],\n        y_true: np.ndarray | pd.Series,\n        participant_ids: np.ndarray | pd.Series,\n    ) -> StackedFusionModel:\n    def fit(self, base_predictions_dict: dict[str, np.ndarray | pd.Series], y_true: np.ndarray | pd.Series, participant_ids: np.ndarray | pd.Series) -> StackedFusionModel:", "def fit(self, base_predictions_dict: dict[str, np.ndarray | pd.Series], y_true: np.ndarray | pd.Series, participant_ids: np.ndarray | pd.Series) -> StackedFusionModel:")
with open("neurosaarthi-ad/models/fusion/stacking.py", "w") as f:
    f.write(content)

with open("neurosaarthi-ad/harmonization/visits.py", "r") as f:
    content = f.read()
content = content.replace("if (\n        visits.groupby(participant_col, sort=False)[order_col]\n        .diff()\n        .fillna(0)\n        .ge(0)\n        .all()\n    ):\n    # Use fully vectorized numpy slice logic to avoid slow O(N) Pandas groupby loop overhead\n    # We must sort by participant_id first to ensure contiguous groups for slice comparisons\n    # NOTE: Do NOT sort by order_col, otherwise we are just reordering the visits and defeating the check!\n    # A stable sort ensures the original row order within each participant group is preserved.\n    sorted_visits = visits.sort_values([participant_col], kind=\"stable\")", "# Use fully vectorized numpy slice logic to avoid slow O(N) Pandas groupby loop overhead\n    # We must sort by participant_id first to ensure contiguous groups for slice comparisons\n    # NOTE: Do NOT sort by order_col, otherwise we are just reordering the visits and defeating the check!\n    # A stable sort ensures the original row order within each participant group is preserved.\n    sorted_visits = visits.sort_values([participant_col], kind=\"stable\")")
content = content.replace("if not visits[time_col].is_monotonic_increasing:\n        sorted_visits = visits.sort_values([participant_col], kind=\"stable\")", "sorted_visits = visits.sort_values([participant_col], kind=\"stable\")")
with open("neurosaarthi-ad/harmonization/visits.py", "w") as f:
    f.write(content)

with open("neurosaarthi-ad/evaluation/explainability.py", "r") as f:
    content = f.read()
content = content.replace("try:\n    from sklearn.inspection import permutation_importance\n\n    HAS_SKLEARN = True\nexcept ImportError:\n    HAS_SKLEARN = False", "try:\n    import importlib.util\n    HAS_SKLEARN = importlib.util.find_spec(\"sklearn\") is not None\nexcept ImportError:\n    HAS_SKLEARN = False")
with open("neurosaarthi-ad/evaluation/explainability.py", "w") as f:
    f.write(content)

with open("neurosaarthi-ad/evaluation/survival_metrics.py", "r") as f:
    content = f.read()
content = content.replace("def _numpy_cindex(\n    event_times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray\n) -> float:", "def _numpy_cindex(\n    event_times: np.ndarray, events: np.ndarray, risk_scores: np.ndarray, chunk_size: int = 2000\n) -> float:")
content = re.sub(r"def _numpy_cindex\(\n    event_times: np\.ndarray,\n    events: np\.ndarray,\n    risk_scores: np\.ndarray,\n    chunk_size: int = 2000,\n\) -> float:\n    \"\"\"Pure numpy fallback for Harrell's C-index with memory-efficient chunking\.\"\"\"\n\n\n", "", content)
with open("neurosaarthi-ad/evaluation/survival_metrics.py", "w") as f:
    f.write(content)

with open("neurosaarthi-ad/models/survival/__init__.py", "r") as f:
    content = f.read()
content = content.replace("from .gradient_boosting import GradientBoostingSurvivalModel", "from .gradient_boosting import GradientBoostingSurvivalModel as GradientBoostingSurvivalModel")
with open("neurosaarthi-ad/models/survival/__init__.py", "w") as f:
    f.write(content)
