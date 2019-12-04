"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
from typing import Any
from typing import Iterable
from typing import List
from typing import NamedTuple

import numpy as np
import sklearn.metrics

from deeplearning.ml4pl.models import epoch
from labm8.py import app
from labm8.py import decorators
from labm8.py import progress


FLAGS = app.FLAGS

app.DEFINE_string(
  "batch_scores_averaging_method",
  "weighted",
  "Selects the averaging method to use when computing recall/precision/F1 "
  "scores. See <https://scikit-learn.org/stable/modules/generated/sklearn"
  ".metrics.f1_score.html>",
)


class Data(NamedTuple):
  graph_ids: List[int]
  data: Any

  @property
  def graph_count(self) -> int:
    return len(self.graph_ids)


class BatchIterator(NamedTuple):
  """A batch iterator"""

  batches: Iterable[Data]
  graph_count: int


class Results(NamedTuple):
  targets: np.array
  predictions: np.array

  # Derived metrics.
  loss: float
  accuracy: float
  precision: float
  recall: float
  f1: float

  @property
  def target_count(self) -> int:
    return self.targets.shape[1]

  @classmethod
  def Create(cls, targets: np.array, predictions: np.array, loss: float = 0):
    """

    Args:
      targets: An array of 1-hot target vectors with
        shape (y_count, y_dimensionality), dtype int32.
      predictions: An array of 1-hot prediction vectors with
        shape (y_count, y_dimensionality), dtype int32.

    Returns:
      A Results instance.
    """
    if targets.shape != predictions.shape:
      raise TypeError(
        f"Expected model to produce targets with shape {targets.shape} but "
        f"instead received predictions with shape {predictions.shape}"
      )

    y_dimensionality = targets.shape[1]
    if y_dimensionality < 2:
      raise TypeError(
        f"Expected label dimensionality > 1, received {y_dimensionality}"
      )

    # Create dense arrays of shape (target_count).
    true_y = np.argmax(targets, axis=1)
    pred_y = np.argmax(predictions, axis=1)

    # NOTE(github.com/ChrisCummins/ProGraML/issues/22): This requires that
    # labels have values [0,...n).
    labels = np.arange(y_dimensionality, dtype=np.int32)

    return cls(
      targets=targets,
      predictions=predictions,
      loss=loss,
      accuracy=sklearn.metrics.accuracy_score(true_y, pred_y),
      precision=sklearn.metrics.precision_score(
        true_y,
        pred_y,
        labels=labels,
        average=FLAGS.batch_scores_averaging_method,
      ),
      recall=sklearn.metrics.recall_score(
        true_y,
        pred_y,
        labels=labels,
        average=FLAGS.batch_scores_averaging_method,
      ),
      f1=sklearn.metrics.f1_score(
        true_y,
        pred_y,
        labels=labels,
        average=FLAGS.batch_scores_averaging_method,
      ),
    )

  def __eq__(self, rhs: "Results"):
    return self.accuracy == rhs.accuracy

  def __gt__(self, rhs: "Results"):
    return self.accuracy > rhs.accuracy


class RollingResults:
  def __init__(self):
    self.batch_count = 0
    self.loss_sum = 0
    self.accuracy_sum = 0
    self.precision_sum = 0
    self.recall_sum = 0
    self.f1_sum = 0

  def Update(self, results: Results):
    self.batch_count += 1
    self.loss_sum += results.loss
    self.accuracy_sum += results.accuracy
    self.precision_sum += results.precision
    self.recall_sum += results.recall
    self.f1_sum += results.f1

  @property
  def loss(self) -> float:
    return self.loss_sum / max(self.batch_count, 1)

  @property
  def accuracy(self) -> float:
    return self.accuracy_sum / max(self.batch_count, 1)

  @property
  def precision(self) -> float:
    return self.precision_sum / max(self.batch_count, 1)

  @property
  def recall(self) -> float:
    return self.recall_sum / max(self.batch_count, 1)

  @property
  def f1(self) -> float:
    return self.f1_sum / max(self.batch_count, 1)

  def ToEpochResults(self) -> epoch.Results:
    return epoch.Results(
      batch_count=self.batch_count,
      loss=self.loss,
      accuracy=self.accuracy,
      precision=self.precision,
      recall=self.recall,
      f1=self.f1,
    )
