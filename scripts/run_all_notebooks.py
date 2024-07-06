#!/usr/bin/env python3
"""This script runs all example notebooks."""
import subprocess
from pathlib import Path

# notebook folders
notebook_folders = ["applications", "algorithms"]

# discover all notebooks
notebooks = []
for folder in notebook_folders:
    notebooks.extend(list(Path(folder).rglob("*.ipynb")))

# run each notebook
last_failed = False
for nb in notebooks:
    subprocess.run(
        ["python", "scripts/run_notebook.py", "-n", str(nb)], capture_output=True
    )

print("All notebooks run successfully.")
