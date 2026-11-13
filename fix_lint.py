with open("neurosaarthi-ad/dashboards/streamlit_app.py", "r") as f:
    lines = f.readlines()
with open("neurosaarthi-ad/dashboards/streamlit_app.py", "w") as f:
    for i, line in enumerate(lines):
        if i >= 202:
            continue
        f.write(line)
