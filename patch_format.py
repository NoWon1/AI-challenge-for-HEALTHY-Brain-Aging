import subprocess
import sys

# Format the file using black/ruff
subprocess.run([sys.executable, "-m", "black", "neurosaarthi-ad/demo/runtime.py"])
