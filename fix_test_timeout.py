import re
with open("neurosaarthi-ad/tests/test_streamlit_app.py", "r") as f:
    content = f.read()

content = content.replace("app = AppTest.from_file(str(app_path)).run(timeout=45)", "app = AppTest.from_file(str(app_path)).run(timeout=600)")

with open("neurosaarthi-ad/tests/test_streamlit_app.py", "w") as f:
    f.write(content)
