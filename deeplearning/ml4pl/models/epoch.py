"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
import enum
from typing import Any
from typing import NamedTuple
from typing import Optional

from labm8.py import app
from labm8.py import shell


FLAGS = app.FLAGS


class Type(enum.Enum):
  TRAIN = 0
  VAL = 1
  TEST = 2


class Results(NamedTuple):
  accuracy: float = 0
  precision: float = 0
  recall: float = 0
  f1: float = 0
  batch_count: int = None
  loss: float = None

  def __repr__(self) -> str:
    return (
      f"accuracy={self.accuracy:.2%}%, "
      f"precision={self.precision:.3f}, recall={self.recall:.3f}, f1={self.f1:.3f}"
    )

  def ToFormattedString(self, previous: Optional["Results"]) -> str:
    previous = previous or self()

    def Colorize(new, old, string):
      if new > old:
        return f"{shell.ShellEscapeCodes.BOLD}{shell.ShellEscapeCodes.GREEN}{string}{shell.ShellEscapeCodes.END}"
      elif new == old:
        return f"{shell.ShellEscapeCodes.BOLD}{shell.ShellEscapeCodes.YELLOW}{string}{shell.ShellEscapeCodes.END}"
      else:
        return f"{shell.ShellEscapeCodes.BOLD}{shell.ShellEscapeCodes.RED}{string}{shell.ShellEscapeCodes.END}"

    accuracy = Colorize(
      self.accuracy, previous.accuracy, f"accuracy={self.accuracy:.2%}"
    )
    precision = Colorize(
      self.precision, previous.precision, f"precision={self.precision:.3f}"
    )
    recall = Colorize(self.recall, previous.recall, f"recall={self.recall:.3f}")
    return f"{accuracy}, {precision}, {recall}"

  def __eq__(self, rhs: "Results"):
    return self.accuracy == rhs.accuracy

  def __gt__(self, rhs: "Results"):
    return self.accuracy > rhs.accuracy


class BestResults(NamedTuple):
  epoch_num: int = 0
  results: Results = Results()

  def __repr__(self):
    return f"{self.results} at epoch {self.epoch_num}"
