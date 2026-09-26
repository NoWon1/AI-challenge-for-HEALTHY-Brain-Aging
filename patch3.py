import sys

with open("neurosaarthi-ad/demo/runtime.py", "r") as f:
    text = f.read()

# Replace baseline definition
old_baseline = """
                    score_frame = pd.DataFrame(index=frame.index)
                    for modality, distribution in modality_distributions.items():
                        values = distribution[bootstrap_index, :, horizon_index].copy()
                        values[~self._available_mask(frame, modality).to_numpy()] = np.nan
                        score_frame[modality] = values
                    fused[bootstrap_index, :, horizon_index] = weighted_score_fusion(
"""
new_baseline = """
                    score_dict = {}
                    for modality, distribution in modality_distributions.items():
                        values = distribution[bootstrap_index, :, horizon_index].copy()
                        values[~self._available_mask(frame, modality).to_numpy()] = np.nan
                        score_dict[modality] = values
                    score_frame = pd.DataFrame(score_dict, index=frame.index)
                    fused[bootstrap_index, :, horizon_index] = weighted_score_fusion(
"""
text = text.replace(old_baseline.strip("\n"), new_baseline.strip("\n"))

# Replace normal distribution
old_dist = """
                score_frame = pd.DataFrame(index=frame.index)
                for modality, distribution in modality_distributions.items():
                    values = distribution[bootstrap_index, :, horizon_index].copy()
                    if modality in disabled:
                        values[:] = np.nan
                    else:
                        values[~self._available_mask(frame, modality).to_numpy()] = np.nan
                    score_frame[modality] = values
                fused[bootstrap_index, :, horizon_index] = weighted_score_fusion(
"""
new_dist = """
                score_dict = {}
                for modality, distribution in modality_distributions.items():
                    values = distribution[bootstrap_index, :, horizon_index].copy()
                    if modality in disabled:
                        values[:] = np.nan
                    else:
                        values[~self._available_mask(frame, modality).to_numpy()] = np.nan
                    score_dict[modality] = values
                score_frame = pd.DataFrame(score_dict, index=frame.index)
                fused[bootstrap_index, :, horizon_index] = weighted_score_fusion(
"""
text = text.replace(old_dist.strip("\n"), new_dist.strip("\n"))

with open("neurosaarthi-ad/demo/runtime.py", "w") as f:
    f.write(text)
