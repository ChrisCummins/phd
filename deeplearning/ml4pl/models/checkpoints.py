"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
from typing import Any
from typing import NamedTuple

from deeplearning.ml4pl import run_id as run_id_lib
from labm8.py import app


FLAGS = app.FLAGS


class Checkpoint(NamedTuple):
  run_id: run_id_lib.RunId
  epoch_num: int
  model_data: Any
