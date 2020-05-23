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
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning

from labm8.py import app
from labm8.py import humanize
from labm8.py import pbutil
from labm8.py import ppar
from programl.ml.batch.async_batch_builder import AsyncBatchBuilder
from programl.ml.model.ggnn.ggnn import Ggnn
from programl.ml.model.ggnn.ggnn_batch_builder import GgnnModelBatchBuilder
from programl.proto import checkpoint_pb2
from programl.proto import epoch_pb2
from programl.task.dataflow import vocabulary
from programl.task.dataflow.graph_loader import DataflowGraphLoader

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


def TrainDataflowGGNN(
  path: pathlib.Path,
  analysis: str,
  limit_max_data_flow_steps: bool,
  train_graph_counts: List[int],
  val_graph_count: int,
  val_seed: int,
  batch_size: int,
  use_cdfg: bool,
  max_vocab_size: int,
  target_vocab_cumfreq: float,
  run_id: Optional[str] = None,
) -> pathlib.Path:
  if not path.is_dir():
    raise FileNotFoundError(path)

  # Since we are dealing with binary classification we calculate
  # precesion / recall / F1 wrt only the positive class.
  FLAGS.batch_results_averaging_method = "binary"
  # NOTE(github.com/ChrisCummins/ProGraML/issues/13): F1 score computation
  # warns that it iss undefined when there are missing instances from a class,
  # which is fine for our usage.
  warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

  # Create the logging directories.
  run_id = run_id or time.strftime("%y:%m:%dT%H:%M:%S")
  model_name = "cdfg" if use_cdfg else "programl"
  log_relpath = f"logs/{model_name}/{analysis}/{run_id}"
  log_dir = path / log_relpath
  if log_dir.is_dir():
    raise OSError(
      f"Logs directory already exists. Refusing to overwrite: {log_dir}"
    )
  app.Log(1, "Writing logs to %s", log_dir)
  log_dir.mkdir(parents=True)
  (log_dir / "epochs").mkdir()
  (log_dir / "checkpoints").mkdir()
  (log_dir / "graph_loader").mkdir()

  RecordExperimentalSetup(log_dir)

  vocab = vocabulary.LoadVocabulary(
    path,
    use_cdfg=use_cdfg,
    max_items=max_vocab_size,
    target_cumfreq=target_vocab_cumfreq,
  )

  # Create the model, defining the shape of the graphs that it will process.
  #
  # For these data flow experiments, our graphs contain per-node binary
  # classification targets (e.g. reachable / not-reachable).
  model = Ggnn(
    vocabulary=vocab,
    test_only=False,
    node_y_dimensionality=2,
    graph_y_dimensionality=0,
    graph_x_dimensionality=0,
    use_selector_embeddings=True,
  )

  # Optionally limit the size of graphs we use.
  if limit_max_data_flow_steps:
    data_flow_step_max = model.message_passing_step_count
  else:
    data_flow_step_max = None

  # Read val batches asynchronously
  val_batches = AsyncBatchBuilder(
    batch_builder=GgnnModelBatchBuilder(
      graph_loader=DataflowGraphLoader(
        path,
        epoch_type=epoch_pb2.VAL,
        analysis=analysis,
        min_graph_count=val_graph_count,
        max_graph_count=val_graph_count,
        data_flow_step_max=data_flow_step_max,
        logfile=open(log_dir / "graph_loader" / "val.txt", "w"),
        seed=val_seed,
        use_cdfg=use_cdfg,
      ),
      vocabulary=vocab,
      max_node_size=batch_size,
    ),
  )
  val_batches.start()

  # Cumulative totals for training graph counts at each "epoch".
  train_graph_cumsums = np.array(train_graph_counts, dtype=np.int32)
  # The number of training graphs in each "epoch".
  train_graph_counts = train_graph_cumsums - np.concatenate(
    ([0], train_graph_counts[:-1])
  )

  for epoch_step, (train_graph_cumsum, train_graph_count) in enumerate(
    zip(train_graph_cumsums, train_graph_counts), start=1
  ):
    start_time = time.time()

    train_batches = ppar.ThreadedIterator(
      GgnnModelBatchBuilder(
        DataflowGraphLoader(
          path,
          epoch_type=epoch_pb2.TRAIN,
          analysis=analysis,
          min_graph_count=train_graph_count,
          max_graph_count=train_graph_count,
          data_flow_step_max=data_flow_step_max,
          logfile=open(
            log_dir / "graph_loader" / f"{epoch_step:03d}.train.txt", "w"
          ),
          use_cdfg=use_cdfg,
        ),
        vocabulary=vocab,
        max_node_size=batch_size,
      ),
      max_queue_size=100,
    )

    train_results = model.RunBatches(
      epoch_pb2.TRAIN,
      train_batches,
      log_prefix=f"Train to {humanize.Commas(train_graph_cumsum)} graphs",
      total_graph_count=train_graph_count,
    )

    # During validation, wait for the batch builder to finish and then
    # iterate over those.
    val_batches.join()
    val_results = model.RunBatches(
      epoch_pb2.VAL,
      val_batches.batches,
      log_prefix=f"Val at {humanize.Commas(train_graph_cumsum)} graphs",
      total_graph_count=val_graph_count,
    )

    # Write the epoch to file as an epoch list.
    # cat *.EpochList.pbtxt > epochs.pbtxt
    epoch = epoch_pb2.EpochList(
      epoch=[
        epoch_pb2.Epoch(
          walltime_seconds=time.time() - start_time,
          epoch_num=epoch_step,
          train_results=train_results,
          val_results=val_results,
        )
      ]
    )
    print(epoch, end="")
    epoch_relpath = f"epochs/{epoch_step:03d}.EpochList.pbtxt"
    checkpoint_relpath = f"checkpoints/{epoch_step:03d}.Checkpoint.pb"
    pbutil.ToFile(epoch, log_dir / epoch_relpath)
    app.Log(1, "Wrote %s/%s", log_relpath, epoch_relpath)
    pbutil.ToFile(model.SaveCheckpoint(), log_dir / checkpoint_relpath)
    app.Log(1, "Wrote %s/%s", log_relpath, checkpoint_relpath)
  return log_dir


def TestDataflowGGNN(
  path: pathlib.Path,
  log_dir: pathlib.Path,
  analysis: str,
  limit_max_data_flow_steps: bool,
  batch_size: int,
  use_cdfg: bool,
  max_vocab_size: int,
  target_vocab_cumfreq: float,
):
  # Since we are dealing with binary classification we calculate
  # precesion / recall / F1 wrt only the positive class.
  FLAGS.batch_results_averaging_method = "binary"
  # NOTE(github.com/ChrisCummins/ProGraML/issues/13): F1 score computation
  # warns that it iss undefined when there are missing instances from a class,
  # which is fine for our usage.
  warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

  RecordExperimentalSetup(log_dir)

  # Create the logging directories.
  assert (log_dir / "epochs").is_dir()
  assert (log_dir / "checkpoints").is_dir()
  assert (log_dir / "graph_loader").is_dir()

  vocab = vocabulary.LoadVocabulary(
    path,
    use_cdfg=use_cdfg,
    max_items=max_vocab_size,
    target_cumfreq=target_vocab_cumfreq,
  )

  # Create the model, defining the shape of the graphs that it will process.
  #
  # For these data flow experiments, our graphs contain per-node binary
  # classification targets (e.g. reachable / not-reachable).
  model = Ggnn(
    vocabulary=vocab,
    test_only=True,
    node_y_dimensionality=2,
    graph_y_dimensionality=0,
    graph_x_dimensionality=0,
    use_selector_embeddings=True,
  )
  restored_epoch, checkpoint = SelectCheckpoint(log_dir)
  app.Log(
    1,
    "Selected checkpoint at epoch %d to restore with val F1 score %.4f",
    restored_epoch.epoch_num,
    restored_epoch.val_results.mean_f1,
  )
  model.RestoreCheckpoint(checkpoint)

  # Optionally limit the size of graphs we use.
  if limit_max_data_flow_steps:
    data_flow_step_max = model.message_passing_step_count
  else:
    data_flow_step_max = None

  batches = GgnnModelBatchBuilder(
    graph_loader=DataflowGraphLoader(
      path,
      epoch_type=epoch_pb2.TEST,
      analysis=analysis,
      data_flow_step_max=data_flow_step_max,
      logfile=open(log_dir / "graph_loader" / "test.txt", "w"),
      use_cdfg=use_cdfg,
    ),
    vocabulary=vocab,
    max_node_size=batch_size,
  )

  start_time = time.time()
  test_results = model.RunBatches(epoch_pb2.TEST, batches, log_prefix="Test")
  epoch = epoch_pb2.EpochList(
    epoch=[
      epoch_pb2.Epoch(
        walltime_seconds=time.time() - start_time,
        epoch_num=restored_epoch.epoch_num,
        test_results=test_results,
      )
    ]
  )
  print(epoch, end="")

  epoch_path = log_dir / "epochs" / "TEST.EpochList.pbtxt"
  pbutil.ToFile(epoch, epoch_path)
  app.Log(1, "Wrote %s", epoch_path)


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
  return epoch.epoch[0], checkpoint
