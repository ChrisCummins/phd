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
"""This module defines classes for representing model checkpoints.

A checkpoint is a snapshot of a model's state at the end of training for a given
epoch.
"""
from typing import Any
from typing import Dict
from typing import NamedTuple

from programl.ml.epoch import epoch


class Checkpoint(NamedTuple):
  """A model checkpoint."""

  epoch_num: int
  best_results: Dict[epoch.EpochType, epoch.BestResults]
  model_data: Any
