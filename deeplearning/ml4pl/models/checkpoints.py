"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
from typing import Any
from typing import Dict
from typing import NamedTuple

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.models import epoch
from labm8.py import app


FLAGS = app.FLAGS


class CheckpointReference(NamedTuple):
  """Reference to a checkpoint."""

  run_id: run_id_lib.RunId
  epoch_num: int

  def __repr__(self):
    return f"{self.run_id}@{self.epoch_num}"

  @classmethod
  def FromString(cls, string: str) -> "CheckpointReference":
    try:
      if "@" in string:
        components = string.split("@")
        assert len(components) == 2
        run_id_string, epoch_num_string = components
        run_id = run_id_lib.RunId.FromString(run_id_string)
        epoch_num = int(epoch_num_string)
      else:
        run_id = run_id_lib.RunId.FromString(string)
        epoch_num = None

      return CheckpointReference(run_id, epoch_num)
    except Exception:
      raise ValueError(f"Invalid run ID and epoch format: {string}")


class Checkpoint(NamedTuple):
  """A model checkpoint."""

  run_id: run_id_lib.RunId
  epoch_num: int
  best_results: Dict[epoch.Type, epoch.BestResults]
  model_data: Any

  def ToCheckpointReference(self) -> CheckpointReference:
    return CheckpointReference(run_id=self.run_id, epoch_num=self.epoch_num)
