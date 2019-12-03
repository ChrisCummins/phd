"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
from typing import Any
from typing import NamedTuple

import numpy as np

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.models import batch
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import log_database
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


class Logger(object):
  def __init__(self, db: log_database.Database):
    self.db = db

  def Save(self, run_id: run_id_lib.RunId, data_to_save: Any) -> None:
    # TODO: raise NotImplementedError
    pass

  def Load(self, run_id: run_id_lib.RunId) -> Any:
    # TODO: raise NotImplementedError
    pass

  def OnBatch(
    self,
    run_id: run_id_lib.RunId,
    epoch_type: epoch.Type,
    results: batch.Results,
  ):
    pass

  def OnEpochEnd(
    self,
    run_id: run_id_lib.RunId,
    epoch_tupe: epoch.Type,
    results: epoch.Results,
  ):
    pass

  @classmethod
  def FromFlags(cls) -> "Logger":
    if not FLAGS.log_db:
      raise app.UsageError("--log_db not set")
    return Logger(FLAGS.log_db())
