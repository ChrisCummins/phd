"""Module to load loop parallelization dataset from the paper:

    ﻿Maramzin, A., Vasiladiotis, C., Lozano, R. C., & Cole, M. (2019).
    “It Looks Like You’re Writing a Parallel Loop” A Machine Learning
    Based Parallelization Assistant. In AI-SEPS.
"""
import pandas as pd

from labm8 import app
from labm8 import bazelutil

FLAGS = app.FLAGS

DATASET_CSV = bazelutil.DataPath('phd/datasets/ppar_metrics/ppar_metrics.csv')


def LoopDatasetToDataFrame() -> pd.DataFrame():
  return pd.read_csv(DATASET_CSV)
