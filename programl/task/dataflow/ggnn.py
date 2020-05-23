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
"""Logic for training and evaluating GGNNs."""
import pathlib
import time
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from labm8.py import app
from labm8.py import humanize
from labm8.py import pbutil
from labm8.py import ppar
from programl.ml.batch.async_batch_builder import AsyncBatchBuilder
from programl.ml.model.ggnn.ggnn import Ggnn
from programl.proto import epoch_pb2
from programl.task.dataflow import dataflow
from programl.task.dataflow.ggnn_batch_builder import DataflowGgnnBatchBuilder
from programl.task.dataflow.graph_loader import DataflowGraphLoader


def TrainDataflowGGNN(
  path: pathlib.Path,
  analysis: str,
  vocab: Dict[str, int],
  limit_max_data_flow_steps: bool,
  train_graph_counts: List[int],
  val_graph_count: int,
  val_seed: int,
  batch_size: int,
  use_cdfg: bool,
  run_id: Optional[str] = None,
) -> pathlib.Path:
  if not path.is_dir():
    raise FileNotFoundError(path)

  # Create the logging directories.
  log_dir, log_relpath = dataflow.CreateLoggingDirectories(
    dataset_root=path,
    model_name="cdfg" if use_cdfg else "programl",
    analysis=analysis,
    run_id=run_id,
  )

  dataflow.PatchWarnings()
  dataflow.RecordExperimentalSetup(log_dir)

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
    batch_builder=DataflowGgnnBatchBuilder(
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
      DataflowGgnnBatchBuilder(
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
  vocab: Dict[str, int],
  limit_max_data_flow_steps: bool,
  batch_size: int,
  use_cdfg: bool,
):
  dataflow.PatchWarnings()
  dataflow.RecordExperimentalSetup(log_dir)

  # Create the logging directories.
  assert (log_dir / "epochs").is_dir()
  assert (log_dir / "checkpoints").is_dir()
  assert (log_dir / "graph_loader").is_dir()

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
  restored_epoch, checkpoint = dataflow.SelectCheckpoint(log_dir)
  model.RestoreCheckpoint(checkpoint)

  # Optionally limit the size of graphs we use.
  if limit_max_data_flow_steps:
    data_flow_step_max = model.message_passing_step_count
  else:
    data_flow_step_max = None

  batches = DataflowGgnnBatchBuilder(
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
