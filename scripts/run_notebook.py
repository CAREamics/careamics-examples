#!/usr/bin/env python3
"""Run a single notebook and update it."""

import argparse
from datetime import datetime

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

# args parser
parser = argparse.ArgumentParser(description="Run a single notebook.")
parser.add_argument("-n", type=str, help="Notebook to run.")

# parse arguments
args = parser.parse_args()
nb = args.n

with open(nb) as ff:
    last_failed = False

    print(f"{datetime.now()} Will run {nb}")
    nb_in = nbformat.read(ff, nbformat.NO_CONVERT)
    n_cells = len(nb_in.cells)

    # run notebook
    ep = ExecutePreprocessor(timeout=10800)  # time-out 3h
    print(f"{datetime.now()} Running {nb}")

    try:
        nb_out = ep.preprocess(nb_in)
    except Exception as e:
        last_failed = True
        print(f"{datetime.now()} Error running {nb}")
        print(e)

    print(f"{datetime.now()} Done running {nb}")

# save notebook
if not last_failed and n_cells == len(nb_out.cells):
    # make sure that the notebook has the same number of cells
    # this can avoid overwriting the notebook with an empty one
    print(f"{datetime.now()} Writing {nb}")
    with open(nb, "w") as ff:
        nbformat.write(nb_out, ff)
