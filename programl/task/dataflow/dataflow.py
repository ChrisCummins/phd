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
"""This module defines functions for training and testing GGNN dataflow models.
"""
import json
import pathlib
import time
import warnings
from typing import Tuple

from sklearn.exceptions import UndefinedMetricWarning

from labm8.py import app
from labm8.py import pbutil
from programl.proto import checkpoint_pb2
from programl.proto import epoch_pb2

FLAGS = app.FLAGS


def RecordExperimentalSetup(log_dir: pathlib.Path) -> None:
  """Create flags.txt and build_info.json files.

  These two files record a snapshot of the configuration and build information,
  useful for debugging and reproducibility.

  Args:
    log_dir: The path to write the files in.
  """
  with open(log_dir / "flags.txt", "w") as f:
    f.write(app.FlagsToString())
  with open(log_dir / "build_info.json", "w") as f:
    json.dump(app.ToJson(), f, sort_keys=True, indent=2, separators=(",", ": "))


def PatchWarnings():
  # Since we are dealing with binary classification we calculate
  # precesion / recall / F1 wrt only the positive class.
  FLAGS.batch_results_averaging_method = "binary"
  # NOTE(github.com/ChrisCummins/ProGraML/issues/13): F1 score computation
  # warns that it is undefined when there are missing instances from a class,
  # which is fine for our usage.
  warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


def SelectCheckpoint(
  log_dir: pathlib.Path,
) -> Tuple[epoch_pb2.Epoch, checkpoint_pb2.Checkpoint]:
  """Select a checkpoint to load.

  Returns:
    A tuple of <Epoch, Checkpoint> messages.
  """
  best_f1 = -1
  best_epoch_num = None
  for path in (log_dir / "epochs").iterdir():
    if path.name.endswith(".EpochList.pbtxt"):
      epoch = pbutil.FromFile(path, epoch_pb2.EpochList())
      f1 = epoch.epoch[0].val_results.mean_f1
      epoch_num = epoch.epoch[0].epoch_num
      if f1 >= best_f1:
        best_f1 = f1
        best_epoch_num = epoch_num
  epoch = pbutil.FromFile(
    log_dir / "epochs" / f"{best_epoch_num:03d}.EpochList.pbtxt",
    epoch_pb2.EpochList(),
  )
  checkpoint = pbutil.FromFile(
    log_dir / "checkpoints" / f"{best_epoch_num:03d}.Checkpoint.pb",
    checkpoint_pb2.Checkpoint(),
  )
  app.Log(
    1,
    "Selected checkpoint at epoch %d to restore with val F1 score %.4f",
    epoch.epoch[0].epoch_num,
    epoch.epoch[0].val_results.mean_f1,
  )
  return epoch.epoch[0], checkpoint


def CreateLoggingDirectories(
  dataset_root: pathlib.Path, model_name: str, analysis: str, run_id: str = None
):
  # Create the logging directories.
  run_id = run_id or time.strftime("%y:%m:%dT%H:%M:%S")
  log_relpath = f"logs/{model_name}/{analysis}/{run_id}"
  log_dir = dataset_root / log_relpath
  if log_dir.is_dir():
    raise OSError(
      f"Logs directory already exists. Refusing to overwrite: {log_dir}"
    )
  app.Log(1, "Writing logs to %s", log_dir)
  log_dir.mkdir(parents=True)
  (log_dir / "epochs").mkdir()
  (log_dir / "checkpoints").mkdir()
  (log_dir / "graph_loader").mkdir()
  return log_dir, log_relpath
