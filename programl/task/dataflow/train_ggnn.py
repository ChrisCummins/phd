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
"""Train a GGNN to estimate solutions for classic data flow problems.

This script reads ProGraML graphs and uses a GGNN to predict binary
classification targets for data flow problems.
"""
import pathlib
import socket
import time
import warnings
from typing import Dict

from sklearn.exceptions import UndefinedMetricWarning

from labm8.py import app
from labm8.py import pbutil
from labm8.py import ppar
from programl.ml.batch.async_batch_builder import AsyncBatchBuilder
from programl.ml.model.ggnn.ggnn import Ggnn
from programl.ml.model.ggnn.ggnn_batch_builder import GgnnModelBatchBuilder
from programl.proto import epoch_pb2
from programl.task.dataflow.graph_loader import DataflowGraphLoader

app.DEFINE_string(
  "path",
  str(pathlib.Path("~/programl/dataflow").expanduser()),
  "The path to read from",
)
app.DEFINE_string("analysis", "reachability", "The analysis type to use.")
app.DEFINE_integer(
  "val_graph_count", 10000, "The number of graphs to use in the validation set."
)
app.DEFINE_integer(
  "val_seed", 0xCC, "The seed value for randomly sampling validation graphs.",
)
app.DEFINE_integer(
  "batch_size",
  50000,
  "The number of nodes in a graph. "
  "On our system, we observed that a batch size of 50,000 nodes requires "
  "about 5.2GB of GPU VRAM.",
)
app.DEFINE_boolean(
  "limit_max_data_flow_steps",
  True,
  "If set, limit the size of dataflow-annotated graphs used to only those with "
  "data_flow_steps <= message_passing_step_count",
)
app.DEFINE_list(
  "train_graph_counts",
  [
    1000,
    2000,
    3000,
    4000,
    5000,
    10000,
    20000,
    30000,
    40000,
    50000,
    100000,
    200000,
    300000,
    400000,
    500000,
    1000000,
  ],
  "The list of cumulative training graph counts to evaluate at.",
)
FLAGS = app.FLAGS


def LoadVocabulary(path: pathlib.Path) -> Dict[str, int]:
  with open(path) as f:
    vocab = f.readlines()
  return {v: i for i, v in enumerate(vocab)}


def Main():
  """Main entry point."""
  # The data directory which we will read model inputs from, and write logs to.
  path = pathlib.Path(FLAGS.path)
  analysis = FLAGS.analysis
  limit_max_data_flow_steps = FLAGS.limit_max_data_flow_steps
  val_graph_count = FLAGS.val_graph_count
  batch_size = FLAGS.batch_size
  train_graph_counts = [int(x) for x in FLAGS.train_graph_counts]
  val_seed = FLAGS.val_seed

  # Since we are dealing with binary classification we calculate
  # precesion / recall / F1 wrt only the positive class.
  FLAGS.batch_results_averaging_method = "binary"
  # NOTE(github.com/ChrisCummins/ProGraML/issues/13): F1 score computation
  # warns that it iss undefined when there are missing instances from a class,
  # which is fine for our usage.
  warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

  # Create the logging directories.
  uid = f"{socket.gethostname()}@{time.strftime('%y:%m:%dT%H:%M:%S')}"
  log_dir = path / "ml" / "logs" / "ggnnn" / analysis / uid
  app.Log(1, "Writing logs to %s", log_dir.absolute())
  log_dir.mkdir(parents=True)
  (log_dir / "epochs").mkdir()
  (log_dir / "checkpoints").mkdir()
  (log_dir / "graph_loader").mkdir()

  vocabulary = LoadVocabulary(path / "vocabulary.txt")

  # Create the model, defining the shape of the graphs that it will process.
  #
  # For these data flow experiments, our graphs contain per-node binary
  # classification targets (e.g. reachable / not-reachable).
  model = Ggnn(
    vocabulary=vocabulary,
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
        max_graph_count=val_graph_count,
        data_flow_step_max=data_flow_step_max,
        logfile=open(log_dir / "graph_loader" / "val.txt", "w"),
        seed=val_seed,
      ),
      vocabulary=vocabulary,
      max_node_size=batch_size,
    ),
  )
  val_batches.start()

  train_graph_count = 0
  for epoch_step, target_train_graph_count in enumerate(
    train_graph_counts, start=1
  ):
    start_time = time.time()
    log_prefix = f"Epoch {epoch_step} of {len(train_graph_counts)}"

    # Read a training "step" worth of graphs.
    train_graphs_in_step = target_train_graph_count - train_graph_count
    graph_loader = DataflowGraphLoader(
      path,
      epoch_type=epoch_pb2.TRAIN,
      analysis=analysis,
      max_graph_count=train_graphs_in_step,
      data_flow_step_max=data_flow_step_max,
      logfile=open(
        log_dir / "graph_loader" / f"{epoch_step:03d}.train.txt", "w"
      ),
    )
    # Construct batches from those graphs in a background thread.
    batch_builder = GgnnModelBatchBuilder(
      graph_loader, vocabulary, max_node_size=batch_size
    )
    train_batches = ppar.ThreadedIterator(batch_builder, max_queue_size=5)

    train_results = model.RunBatches(epoch_pb2.TRAIN, train_batches, log_prefix)

    # During validation, wait for the batch builder to finish and then
    # iterate over those.
    val_batches.join()
    val_results = model.RunBatches(
      epoch_pb2.VAL, val_batches.batches, log_prefix
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
    pbutil.ToFile(
      epoch, log_dir / "epochs" / f"{epoch_step:03d}.EpochList.pbtxt"
    )
    pbutil.ToFile(
      model.SaveCheckpoint(),
      log_dir / "checkpoints" / f"{epoch_step:03d}.Checkpoint.pb",
    )


if __name__ == "__main__":
  app.Run(Main)
