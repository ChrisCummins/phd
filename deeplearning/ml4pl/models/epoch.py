"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
import enum
from typing import NamedTuple
from typing import Optional

from deeplearning.ml4pl.models import batch
from labm8.py import app
from labm8.py import shell


FLAGS = app.FLAGS


class Type(enum.Enum):
  TRAIN = 0
  VAL = 1
  TEST = 2


class Results(NamedTuple):
  """The results of an epoch."""

  # The number of batches in the epoch.
  batch_count: int = 0
  # The number of graphs in the epoch.
  graph_count: int = 0
  # The number of targets in the epoch.
  target_count: int = 0
  # The average iteration count of the model.
  iteration_count: float = 0
  # The ratio of batches that resulted in the model converging.
  model_converged: float = 1
  # The average model learning rate, if applicable.
  learning_rate: Optional[float] = None
  # The average model loss, if applicable.
  loss: Optional[float] = None
  # The average accuracy across all predictions.
  accuracy: float = 0
  # The average precision across all predictions.
  precision: float = 0
  # The average recall across all predictions.
  recall: float = 0
  # The average f1 score across all predictions.
  f1: float = 0

  @property
  def has_learning_rate(self) -> bool:
    return self.learning_rate is not None

  @property
  def has_loss(self) -> bool:
    return self.loss is not None

  def __repr__(self) -> str:
    return (
      f"accuracy={self.accuracy:.2%}%, "
      f"precision={self.precision:.3f}, recall={self.recall:.3f}, f1={self.f1:.3f}"
    )

  def ToFormattedString(self, previous: Optional["Results"]) -> str:
    previous: Results = previous or self()

    def Colorize(new, old, string):
      if new >= old:
        return f"{shell.ShellEscapeCodes.BOLD}{shell.ShellEscapeCodes.GREEN}{string}{shell.ShellEscapeCodes.END}"
      else:
        return f"{shell.ShellEscapeCodes.BOLD}{shell.ShellEscapeCodes.RED}{string}{shell.ShellEscapeCodes.END}"

    strings = [
      Colorize(
        self.accuracy, previous.accuracy, f"accuracy={self.accuracy:.2%}"
      ),
      Colorize(
        self.precision, previous.precision, f"precision={self.precision:.3f}"
      ),
      Colorize(self.recall, previous.recall, f"recall={self.recall:.3f}"),
    ]

    if self.has_loss:
      strings.append(
        Colorize(self.loss, previous.loss or 0, f"loss={self.loss:.6f}")
      )

    return ", ".join(strings)

  @classmethod
  def FromRollingResults(
    cls, rolling_results: batch.RollingResults
  ) -> "Results":
    return cls(
      batch_count=rolling_results.batch_count,
      graph_count=rolling_results.graph_count,
      target_count=rolling_results.target_count,
      iteration_count=rolling_results.iteration_count,
      model_converged=rolling_results.model_converged,
      learning_rate=rolling_results.learning_rate,
      loss=rolling_results.loss,
      accuracy=rolling_results.accuracy,
      precision=rolling_results.precision,
      recall=rolling_results.recall,
      f1=rolling_results.f1,
    )

  def __eq__(self, rhs: "Results"):
    return self.accuracy == rhs.accuracy

  def __gt__(self, rhs: "Results"):
    return self.accuracy > rhs.accuracy


class BestResults(NamedTuple):
  epoch_num: int = 0
  results: Results = Results()

  def __repr__(self):
    return f"{self.results} at epoch {self.epoch_num}"
