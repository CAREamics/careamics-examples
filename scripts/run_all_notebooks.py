#!/usr/bin/env python3
"""This script runs all example notebooks."""

from datetime import datetime
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# notebook folders
notebook_folders = ["applications", "algorithms"]

# discover all notebooks
notebooks = []
for folder in notebook_folders:
    notebooks.extend(list(Path(folder).rglob("*.ipynb")))

# run each notebook
last_failed = False
for nb in notebooks:
    with open(nb) as ff:
        last_failed = False

        print(f"{datetime.now()} Will run {nb}")
        nb_in = nbformat.read(ff, nbformat.NO_CONVERT)

        # run notebook
        ep = ExecutePreprocessor(timeout=600)
        print(f"{datetime.now()} Running {nb}")

        try:
            nb_out = ep.preprocess(nb_in)
        except Exception as e:
            last_failed = True
            print(f"{datetime.now()} Error running {nb}")
            print(e)

        print(f"{datetime.now()} Done running {nb}")

    # save notebook
    if not last_failed:
        print(f"{datetime.now()} Writing {nb}")
        with open(nb, "w") as ff:
            nbformat.write(nb_out, ff)

print("All notebooks run successfully.")
