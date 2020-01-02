# Copyright 2019 the ProGraML authors.
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
"""This module defines classes for representing model checkpoints.

A checkpoint is a snapshot of a model's state at the end of training for a given
epoch.
"""
from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import Optional

from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.models import epoch
from labm8.py import app


FLAGS = app.FLAGS


class CheckpointReference(NamedTuple):
  """Reference to a checkpoint."""

  run_id: run_id_lib.RunId
  # If epoch_num is None, then load the best epoch.
  epoch_num: Optional[int]

  def __repr__(self):
    return (
      f"{self.run_id}@{'best' if self.epoch_num is None else self.epoch_num}"
    )

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
    """Return the checkpoint reference for a given checkpoint.

    Use this when you want to keep a reference to a checkpoint, but don't need
    to store the underlying model data, which may be large.
    """
    return CheckpointReference(run_id=self.run_id, epoch_num=self.epoch_num)
