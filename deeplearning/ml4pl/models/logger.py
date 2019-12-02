"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
from typing import NamedTuple

import numpy as np

from labm8.py import app


FLAGS = app.FLAGS


app.DEFINE_string(
  "batch_scores_averaging_method",
  "weighted",
  "Selects the averaging method to use when computing recall/precision/F1 "
  "scores. See <https://scikit-learn.org/stable/modules/generated/sklearn"
  ".metrics.f1_score.html>",
)

app.DEFINE_list(
  "batch_log_types",
  ["val", "test"],
  "The types of epochs to record per-instance batch logs for.",
)


# The result of running a minibatch. Return 1-hot target values and the raw
# 1-hot outputs of the model. These are used to compute evaluation metrics.
class MinibatchResults(NamedTuple):
  y_true_1hot: np.array  # Shape [num_labels,num_classes]
  y_pred_1hot: np.array  # Shape [num_labels,num_classes]
