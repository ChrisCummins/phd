"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
from typing import Any
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional

import numpy as np
import sklearn.metrics

from labm8.py import app


FLAGS = app.FLAGS

app.DEFINE_string(
  "batch_scores_averaging_method",
  "weighted",
  "Selects the averaging method to use when computing recall/precision/F1 "
  "scores. See <https://scikit-learn.org/stable/modules/generated/sklearn"
  ".metrics.f1_score.html>",
)


class Data(NamedTuple):
  """The model data for a batch."""

  graph_ids: List[int]
  data: Any
  # A flag used to mark that this batch is the end of an iterable sequences of
  # batches.
  end_of_batches: bool = False

  @property
  def graph_count(self) -> int:
    return len(self.graph_ids)


def EmptyBatch() -> Data:
  """Construct an empty batch."""
  return Data(graph_ids=[], data=None)


def EndOfBatches() -> Data:
  """Construct a 'end of batches' marker."""
  return Data(graph_ids=[], data=None, end_of_batches=True)


class BatchIterator(NamedTuple):
  """A batch iterator"""

  batches: Iterable[Data]
  # The total number of graphs in all of the batches.
  graph_count: int


class Results(NamedTuple):
  """The results of running a batch through a model.

  Don't instantiate this tuple directly, use Results.Create().
  """

  targets: np.array
  predictions: np.array
  # The number of model iterations to compute the final results. This is used
  # by iterative models such as message passing networks.
  iteration_count: int
  # For iterative models, this indicates whether the state of the model at
  # iteration_count had converged on a solution.
  model_converged: bool
  # The learning rate and loss of models, if applicable.
  learning_rate: Optional[float]
  loss: Optional[float]
  # Batch-level average performance metrics.
  accuracy: float
  precision: float
  recall: float
  f1: float

  @property
  def has_learning_rate(self) -> bool:
    return self.learning_rate is not None

  @property
  def has_loss(self) -> bool:
    return self.loss is not None

  @property
  def target_count(self) -> int:
    """Get the number of targets in the batch.

    For graph-level classifiers, this will be equal to Data.graph_count, else
    it's equal to the batch node count.
    """
    return self.targets.shape[1]

  def __repr__(self) -> str:
    return (
      f"accuracy={self.accuracy:.2%}%, "
      f"precision={self.precision:.3f}, "
      f"recall={self.recall:.3f}, "
      f"f1={self.f1:.3f}"
    )

  def __eq__(self, rhs: "Results"):
    """Compare batch results."""
    return self.accuracy == rhs.accuracy

  def __gt__(self, rhs: "Results"):
    """Compare batch results."""
    return self.accuracy > rhs.accuracy

  @classmethod
  def Create(
    cls,
    targets: np.array,
    predictions: np.array,
    iteration_count: int = 1,
    model_converged: bool = True,
    learning_rate: Optional[float] = None,
    loss: Optional[float] = None,
  ):
    """Construct a results instance from 1-hot targets and predictions.

    This is the preferred means of construct a Results instance, which takes
    care of evaluating all of the metrics for you. The behavior of metrics
    calculation is dependent on the --batch_scores_averaging_method flag.

    Args:
      targets: An array of 1-hot target vectors with
        shape (y_count, y_dimensionality), dtype int32.
      predictions: An array of 1-hot prediction vectors with
        shape (y_count, y_dimensionality), dtype int32.
      iteration_count: For iterative models, the number of model iterations to
        compute the final result.
      model_converged: For iterative models, whether model converged.
      learning_rate: The model learning rate, if applicable.
      loss: The model loss, if applicable.

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

    # NOTE(github.com/ChrisCummins/ProGraML/issues/22): This assumes that
    # labels use the values [0,...n).
    labels = np.arange(y_dimensionality, dtype=np.int64)

    return cls(
      targets=targets,
      predictions=predictions,
      iteration_count=iteration_count,
      model_converged=model_converged,
      learning_rate=learning_rate,
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


class RollingResults:
  """Maintain weighted rolling averages across batches."""

  def __init__(self):
    self.weight_sum = 0
    self.batch_count = 0
    self.graph_count = 0
    self.target_count = 0
    self.weighted_iteration_count_sum = 0
    self.weighted_model_converged_sum = 0
    self.has_learning_rate = False
    self.weighted_learning_rate_sum = 0
    self.has_loss = False
    self.weighted_loss_sum = 0
    self.weighted_accuracy_sum = 0
    self.weighted_precision_sum = 0
    self.weighted_recall_sum = 0
    self.weighted_f1_sum = 0

  def Update(
    self, data: Data, results: Results, weight: Optional[float] = None
  ) -> None:
    """Update the rolling results with a new batch.

    Args:
      data: The batch data used to produce the results.
      results: The batch results to update the current state with.
      weight: A weight to assign to weighted sums. E.g. to weight results
        across all targets, use weight=results.target_count. To weight across
        targets, use weight=batch.target_count. To weight across
        graphs, use weight=batch.graph_count. By default, weight by target
        count.
    """
    if weight is None:
      weight = results.target_count

    self.weight_sum += weight
    self.batch_count += 1
    self.graph_count += data.graph_count
    self.target_count += results.target_count
    self.weighted_iteration_count_sum += results.iteration_count * weight
    self.weighted_model_converged_sum += (
      weight if results.model_converged else 0
    )
    if results.has_learning_rate:
      self.has_learning_rate = True
      self.weighted_learning_rate_sum += results.learning_rate * weight
    if results.has_loss:
      self.has_loss = True
      self.weighted_loss_sum += results.loss * weight
    self.weighted_accuracy_sum += results.accuracy * weight
    self.weighted_precision_sum += results.precision * weight
    self.weighted_recall_sum += results.recall * weight
    self.weighted_f1_sum += results.f1 * weight

  @property
  def iteration_count(self) -> float:
    return self.weighted_iteration_count_sum / max(self.weight_sum, 1)

  @property
  def model_converged(self) -> float:
    return self.weighted_model_converged_sum / max(self.weight_sum, 1)

  @property
  def learning_rate(self) -> Optional[float]:
    if self.has_learning_rate:
      return self.weighted_learning_rate_sum / max(self.weight_sum, 1)

  @property
  def loss(self) -> Optional[float]:
    if self.has_loss:
      return self.weighted_loss_sum / max(self.weight_sum, 1)

  @property
  def accuracy(self) -> float:
    return self.weighted_accuracy_sum / max(self.weight_sum, 1)

  @property
  def precision(self) -> float:
    return self.weighted_precision_sum / max(self.weight_sum, 1)

  @property
  def recall(self) -> float:
    return self.weighted_recall_sum / max(self.weight_sum, 1)

  @property
  def f1(self) -> float:
    return self.weighted_f1_sum / max(self.weight_sum, 1)
