import sys

with open("neurosaarthi-ad/demo/runtime.py", "r") as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "score_frame = pd.DataFrame(score_dict, index=frame.index)" in line:
        if line.startswith("                        score_frame = pd.DataFrame"):
            # Skip the deeply indented one that causes syntax error
            continue
    new_lines.append(line)

with open("neurosaarthi-ad/demo/runtime.py", "w") as f:
    f.writelines(new_lines)
