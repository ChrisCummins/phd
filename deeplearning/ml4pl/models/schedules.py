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
"""This module defines helper enums for controlling runtime behaviour."""
import enum

from labm8.py import app


FLAGS = app.FLAGS


class SaveOn(enum.Enum):
  """Determine when to save checkpoints."""

  # Never save a checkpoint.
  NONE = 0
  # Make a checkpoint at every epoch.
  EVERY_EPOCH = 1
  # Make a checkpoint when validation accuracy improves.
  VAL_IMPROVED = 2


class KeepCheckpoints(enum.Enum):
  """Determine how many checkpoints to keep."""

  # Keep all checkpoints.
  ALL = 1
  # Keep only the last saved checkpoint.
  LAST = 2


class KeepDetailedBatches(enum.Enum):
  """Determine how many detailed batches to keep."""

  # Keep no detailed batches.
  NONE = 0
  # Keep all detailed batches.
  ALL = 1
  # Keep detailed batches only for the last epoch.
  LAST_EPOCH = 2
