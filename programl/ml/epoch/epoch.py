# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module defines a container for storing epoch results."""
import enum
from typing import NamedTuple
from typing import Optional

from labm8.py import shell
from programl.ml.batch.rolling_results import RollingResults


class EpochType(enum.Enum):
  TRAIN = 0
  VAL = 1
  TEST = 2


class EpochResults(NamedTuple):
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
    string = (
      f"accuracy={self.accuracy:.2%}%, "
      f"precision={self.precision:.3f}, recall={self.recall:.3f}, "
      f"f1={self.f1:.3f}"
    )
    if self.has_loss:
      string = f"{string}, loss={self.loss:0.6f}"

    return string

  def ToFormattedString(self, previous: Optional["EpochResults"]) -> str:
    previous: EpochResults = previous or self()

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
  def FromRollingResults(cls, results: RollingResults) -> "EpochResults":
    return cls(
      batch_count=results.batch_count,
      graph_count=results.graph_count,
      target_count=results.target_count,
      iteration_count=results.iteration_count,
      model_converged=results.model_converged,
      learning_rate=results.learning_rate,
      loss=results.loss,
      accuracy=results.accuracy,
      precision=results.precision,
      recall=results.recall,
      f1=results.f1,
    )

  def __eq__(self, rhs: "EpochResults"):
    return self.accuracy == rhs.accuracy

  def __gt__(self, rhs: "EpochResults"):
    return self.accuracy > rhs.accuracy


class BestResults(NamedTuple):
  epoch_num: int = 0
  results: EpochResults = EpochResults()

  def __repr__(self):
    return f"{self.results} at epoch {self.epoch_num}"
