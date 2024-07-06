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
    notebooks.append(list(Path(folder).rglob("*.ipynb")))

# run each notebook
for nb in notebooks:
    with open(nb) as ff:
        print(f"{datetime.now()} Will run {nb}")
        nb_in = nbformat.read(ff, nbformat.NO_CONVERT)
        n_cells = len(nb_in.cells)

        # run notebook
        ep = ExecutePreprocessor(timeout=10800)
        print(f"{datetime.now()} Running {nb}")
        nb_out = ep.preprocess(nb_in)

        print(f"{datetime.now()} Done running {nb}")

    # save notebook
    if n_cells == len(nb_out.cells):
        # make sure that the notebook has the same number of cells
        # this can avoid overwriting the notebook with an empty one
        print(f"{datetime.now()} Writing {nb}")
        with open(nb, "w") as ff:
            nbformat.write(nb_out, ff)

print("All notebooks run successfully.")
