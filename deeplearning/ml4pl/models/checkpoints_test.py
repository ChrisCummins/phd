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
"""Unit tests for //deeplearning/ml4pl/models:checkpoints."""
from deeplearning.ml4pl import run_id as run_id_lib
from deeplearning.ml4pl.models import checkpoints
from labm8.py import test


FLAGS = test.FLAGS


def CheckpointReference_without_epoch_num():
  """Check construction of a checkpoint reference without epoch number."""
  run_id = run_id_lib.RunId.GenerateUnique("reftest")

  a = checkpoints.CheckpointReference(run_id, epoch_num=None)
  assert a.run_id == run_id
  assert a.epoch_num is None

  b = checkpoints.CheckpointReference.FromString(str(a))
  assert b.run_id == run_id
  assert b.epoch_num is None
  assert a == b