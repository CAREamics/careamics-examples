#!/usr/bin/env python3
"""This script runs all example notebooks."""

from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# notebook folders
notebook_folders = ["applications", "algorithms"]

# discover all notebooks
notebooks = []
for folder in notebook_folders:
    notebooks.append(list(Path(folder).rglob("*.ipynb")))

# run each notebook
for nb in notebooks:
    with open(nb) as ff:
        print(f"Will run {nb}")
        nb_in = nbformat.read(ff, nbformat.NO_CONVERT)

        # run notebook
        ep = ExecutePreprocessor(timeout=600, kernel_name="careamics")
        print(f"Running {nb}")
        nb_out = ep.preprocess(nb_in)

    # save notebook
    with open(nb, "w") as ff:
        nbformat.write(nb_out, ff)

print("All notebooks run successfully.")
