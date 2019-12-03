"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
from typing import Any
from typing import List
from typing import NamedTuple

import numpy as np

from labm8.py import app


FLAGS = app.FLAGS


class Data(NamedTuple):
  graph_ids: List[int]
  data: Any


class Results(NamedTuple):
  true_y: np.array
  pred_y: np.array

  # TODO:
  # accuracy: float
  # precisison: float
  # recall: float
  # loss: float = None

  def __eq__(self, rhs: "Results"):
    return self.accuracy == rhs.accuracy

  def __gt__(self, rhs: "Results"):
    return self.accuracy > rhs.accuracy

  @classmethod
  def NullResults(cls):
    return cls([], [])
