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
import sys
import time
import warnings
from typing import Dict

from sklearn.exceptions import UndefinedMetricWarning

from labm8.py import app
from labm8.py import pbutil
from labm8.py import ppar
from programl.ml.batch.async_batch_builder import AsyncBatchBuilder
from programl.ml.batch.rolling_results import RollingResults
from programl.ml.model.ggnn.ggnn import Ggnn
from programl.ml.model.ggnn.ggnn_batch_builder import GgnnModelBatchBuilder
from programl.proto import epoch_pb2
from programl.task.dataflow import graph_loader

app.DEFINE_string(
  "path",
  str(pathlib.Path("~/programl/dataflow").expanduser()),
  "The path to read from",
)
app.DEFINE_string("analysis", "reachability", "The analysis type to use.")
app.DEFINE_integer(
  "max_training_graphs", 1000000, "The maximum number of graphs to train on."
)
app.DEFINE_integer(
  "train_graphs_per_step", 10000, "The number of graphs to train on per step."
)
app.DEFINE_integer(
  "val_graphs", 10000, "The number of graphs to use in the validation set."
)
app.DEFINE_integer("batch_size", 10000, "The number of nodes in a graph.")
app.DEFINE_boolean(
  "limit_max_data_flow_steps",
  True,
  "If set, limit the size of dataflow-annotated graphs used to only those with "
  "data_flow_steps <= message_passing_step_count",
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
  train_graphs_per_step = FLAGS.train_graphs_per_step
  val_graphs = FLAGS.val_graphs
  batch_size = FLAGS.batch_size
  max_training_graphs = FLAGS.max_training_graphs

  # Since we are dealing with binary classification we calculate
  # precesion / recall / F1 wrt only the positive class.
  FLAGS.batch_results_averaging_method = "binary"

  # NOTE(github.com/ChrisCummins/ProGraML/issues/13): F1 score computation
  # warns that it iss undefined when there are missing instances from a class,
  # which is fine for our usage.
  warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

  # Create the logging directories.
  uid = f"{socket.gethostname()}@{time.strftime('%y:%m:%dT%H:%M:%S')}"
  log_dir = path / "ml" / "logs" / "ggnnn" / uid
  app.Log(1, "Writing logs to %s", log_dir.absolute())
  log_dir.mkdir(parents=True)
  (log_dir / "epochs").mkdir()
  (log_dir / "checkpoints").mkdir()

  vocabulary = LoadVocabulary(path / "vocabulary.txt")

  trained_graphs = 0

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
      graph_loader=graph_loader.DataflowGraphLoader(
        path,
        epoch_type=epoch_pb2.VAL,
        analysis=analysis,
        max_graph_count=val_graphs,
        data_flow_step_max=data_flow_step_max,
      ),
      vocabulary=vocabulary,
      max_node_size=batch_size,
    ),
  )
  val_batches.start()

  epoch_step = 0
  batch_step = 0
  while trained_graphs < max_training_graphs:
    start_time = time.time()

    epoch_step += 1
    epoch_results = []
    for epoch_type in [epoch_pb2.TRAIN, epoch_pb2.VAL]:
      if epoch_type == epoch_pb2.TRAIN:
        # Read a training "step" worth of graphs.
        data_loader = graph_loader.DataflowGraphLoader(
          path,
          epoch_type,
          analysis,
          max_graph_count=train_graphs_per_step,
          data_flow_step_max=data_flow_step_max,
        )
        # Construct batches from those graphs in a background thread.
        batch_builder = GgnnModelBatchBuilder(
          data_loader, vocabulary, max_node_size=batch_size
        )
        batches = ppar.ThreadedIterator(batch_builder, max_queue_size=5)
      else:
        # During validation, wait for the batch builder to finish and then
        # iterate over those.
        val_batches.join()
        batches = val_batches.batches

      # Feed the batches through the model and update stats.
      rolling_results = RollingResults()
      for batch_data in batches:
        batch_step += 1
        trained_graphs += batch_data.graph_count
        batch_results = model.RunBatch(epoch_type, batch_data)
        rolling_results.Update(batch_data, batch_results, weight=None)
        print(
          f"\r\033[KEpoch {epoch_step} "
          f"{epoch_pb2.EpochType.Name(epoch_type).lower()}: "
          f"{rolling_results}",
          end="",
          file=sys.stderr,
        )
      print("", file=sys.stderr)

      epoch_results.append(rolling_results.ToEpochResults())

    epoch = epoch_pb2.EpochList(
      epoch=[
        epoch_pb2.Epoch(
          walltime_seconds=time.time() - start_time,
          epoch_num=epoch_step,
          train_results=epoch_results[0],
          val_results=epoch_results[1],
        )
      ]
    )
    print(epoch, end="")
    # Write the epoch to file as an epoch list.
    # cat *.EpochList.pbtxt > epochs.pbtxt
    pbutil.ToFile(
      epoch, log_dir / "epochs" / f"{epoch_step:03d}.EpochList.pbtxt"
    )
    pbutil.ToFile(
      model.SaveCheckpoint(),
      log_dir / "checkpoints" / f"{epoch_step:03d}.Checkpoint.pb",
    )


if __name__ == "__main__":
  app.Run(Main)
