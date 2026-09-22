import re

with open("neurosaarthi-ad/demo/runtime.py", "r") as f:
    content = f.read()

search = """    def _trajectory_for_profile(
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
        widths_array = np.asarray(widths)"""
replace = """    def _trajectory_for_profile(
        self, profile_frame: pd.DataFrame, twin_trajectories: pd.DataFrame
    ) -> pd.DataFrame:
        rows = [{**profile_frame.iloc[0].to_dict(), "horizon_years": float(year)} for year in (1, 2, 3, 5)]
        future = pd.DataFrame(rows)
        centers = self.trajectory_model.predict(future).to_numpy()
        years = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
        centers = np.concatenate([[float(profile_frame.iloc[0]["cognitive_score"])], centers])
        widths = [0.25]

        # ⚡ Bolt: Vectorized numpy splitting and interpolation avoids massive O(N*M) groupby loop overhead
        ordered_twins = twin_trajectories.sort_values(["participant_id", "year"])
        p_ids = ordered_twins["participant_id"].values
        if len(p_ids) > 0:
            split_idx = np.where(p_ids[:-1] != p_ids[1:])[0] + 1
            years_split = np.split(ordered_twins["year"].values, split_idx)
            scores_split = np.split(ordered_twins["cognitive_score"].values, split_idx)
            all_vals = np.array([np.interp(years[1:], y, s) for y, s in zip(years_split, scores_split)])
            dispersions = np.std(all_vals, axis=0) if len(all_vals) > 1 else np.zeros(len(years) - 1)
        else:
            dispersions = np.zeros(len(years) - 1)

        for twin_dispersion in dispersions:
            widths.append(float(np.sqrt(self.trajectory_residual_band**2 + (0.35 * twin_dispersion) ** 2)))

        centers = np.clip(centers, 0.0, 30.0)
        widths_array = np.asarray(widths)"""

search2 = """    def _build_missingness(self) -> pd.DataFrame:
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
        return pd.DataFrame(rows)"""

replace2 = """    def _build_missingness(self) -> pd.DataFrame:
        # ⚡ Bolt: Vectorized missingness computation to avoid slow O(N) groupby loop
        rows = []
        for modality, features in MODALITY_FEATURES.items():
            mask = self.baseline[features].notna().any(axis=1)
            means = mask.groupby(self.baseline["cohort"], sort=False).mean()
            for cohort, mean_val in means.items():
                rows.append({
                    "cohort": cohort,
                    "modality": modality,
                    "missing_rate": float(1.0 - mean_val)
                })
        return pd.DataFrame(rows)"""

if search in content and search2 in content:
    with open("neurosaarthi-ad/demo/runtime.py", "w") as f:
        f.write(content.replace(search, replace).replace(search2, replace2))
    print("Replaced runtime.py")
else:
    print("Not found runtime")
