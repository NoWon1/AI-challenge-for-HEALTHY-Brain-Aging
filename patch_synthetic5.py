import re

with open("neurosaarthi-ad/demo/synthetic.py", "r") as f:
    content = f.read()

# Replace any remaining age variables with base.age
content = content.replace('"age_at_visit": age + max(year_offset, 0.0),', '"age_at_visit": base.age + max(year_offset, 0.0),')

with open("neurosaarthi-ad/demo/synthetic.py", "w") as f:
    f.write(content)
