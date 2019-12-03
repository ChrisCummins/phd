"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
import enum
from typing import NamedTuple

from labm8.py import app


FLAGS = app.FLAGS


class Type(enum.Enum):
  TRAIN = 0
  VAL = 1
  TEST = 2


class Results(NamedTuple):
  accuracy: float
  precision: float
  recall: float
  batch_count: int = None
  loss: float = None

  def __repr__(self) -> str:
    return (
      f"accuracy={self.accuracy:.2%}%, "
      f"precision={self.precision:.3f}, recall={self.recall:.3f}"
    )

  def __eq__(self, rhs: "Results"):
    return self.accuracy == rhs.accuracy

  def __gt__(self, rhs: "Results"):
    return self.accuracy > rhs.accuracy

  @classmethod
  def NullResults(cls):
    return cls(accuracy=0, precision=0, recall=0)
