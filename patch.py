import sys

with open("neurosaarthi-ad/demo/runtime.py", "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if "score_frame = pd.DataFrame(index=frame.index)" in line:
        indent = line[:line.find("score_frame")]
        new_lines.append(f"{indent}score_dict = {{}}\n")
    elif "score_frame[modality] = values" in line:
        indent = line[:line.find("score_frame")]
        new_lines.append(f"{indent}score_dict[modality] = values\n")
        # We need to create score_frame AFTER the loop finishes
    elif "fused[bootstrap_index, :, horizon_index] = weighted_score_fusion(" in line:
        indent = line[:line.find("fused[bootstrap_index")]
        new_lines.append(f"{indent}score_frame = pd.DataFrame(score_dict, index=frame.index)\n")
        new_lines.append(line)
    else:
        new_lines.append(line)

with open("neurosaarthi-ad/demo/runtime.py", "w") as f:
    f.writelines(new_lines)
