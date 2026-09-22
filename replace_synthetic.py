import re

with open("neurosaarthi-ad/demo/synthetic.py", "r") as f:
    content = f.read()

search = """    summary_rows = []
    for cohort, group in baseline.groupby("cohort", sort=False):
        summary_rows.append(
            {
                "cohort": cohort,
                "participants": len(group),
                "role": "Development" if cohort in PUBLIC_COHORTS else ("India adaptation" if cohort == "TLSA" else "External India validation"),
                "setting": _participant_setting(cohort),
                "five_year_event_rate": float(group["event_by_5y"].mean()),
            }
        )"""

replace = """    # ⚡ Bolt: Vectorized groupby aggregation avoids slow O(N) Python loops
    grouped = baseline.groupby("cohort", sort=False)
    sizes = grouped.size()
    rates = grouped["event_by_5y"].mean()

    summary_rows = [
        {
            "cohort": cohort,
            "participants": int(sizes[cohort]),
            "role": "Development" if cohort in PUBLIC_COHORTS else ("India adaptation" if cohort == "TLSA" else "External India validation"),
            "setting": _participant_setting(cohort),
            "five_year_event_rate": float(rates[cohort]),
        }
        for cohort in sizes.index
    ]"""

if search in content:
    with open("neurosaarthi-ad/demo/synthetic.py", "w") as f:
        f.write(content.replace(search, replace))
    print("Replaced synthetic.py")
else:
    print("Not found synthetic")
