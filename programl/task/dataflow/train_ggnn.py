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
"""Run script for machine learning models.

This defines the schedules for running training / validation / testing loops
of a machine learning model.
"""
import pathlib
import random
import time
from typing import Iterable
from typing import Tuple

from labm8.py import app
from labm8.py import pbutil
from labm8.py import ppar
from programl.graph.py import graph_tuple_builder
from programl.ml.batch.batch_data import BatchData
from programl.ml.batch.rolling_results import RollingResults
from programl.ml.model.ggnn.ggnn import Ggnn
from programl.proto import epoch_pb2
from programl.proto import program_graph_features_pb2
from programl.proto import program_graph_pb2


app.DEFINE_string(
  "path",
  str(pathlib.Path("~/programl/dataflow").expanduser()),
  "The path to read from",
)
FLAGS = app.FLAGS


class DataflowGraphLoader(object):
  def __init__(self, path: pathlib.Path, analysis: str):
    self.path = path
    self.labels_path = self.path / analysis
    assert self.labels_path.is_dir()

    self.cache = {
      epoch_pb2.TRAIN: [],
      epoch_pb2.VAL: [],
      epoch_pb2.TEST: [],
    }

  def LoadGraphs(self, epoch_type: epoch_pb2.EpochType):
    dir = self.path / epoch_pb2.EpochType.Name(epoch_type).lower()
    assert dir.is_dir()

    if self.cache[epoch_type]:
      app.Log(1, "Using cache")
      random.shuffle(self.cache[epoch_type])
      for graph, features_list in self.cache[epoch_type]:
        for features in features_list.graph:
          yield graph, features
    else:
      app.Log(1, "Reading from filesystem")
      files = list(dir.iterdir())
      random.shuffle(files)

      for path in files[:1000]:
        stem = path.name[: -len("ProgramGraph.pb")]
        name = f"{stem}ProgramGraphFeaturesList.pb"
        features_path = self.labels_path / name
        if features_path.is_file():
          graph = pbutil.FromFile(path, program_graph_pb2.ProgramGraph())
          features_list = pbutil.FromFile(
            features_path, program_graph_features_pb2.ProgramGraphFeaturesList()
          )
          self.cache[epoch_type].append((graph, features_list))
          for features in features_list.graph:
            yield graph, features


class DataflowGraphBatcher(graph_tuple_builder.GraphTupleBuilder):
  def __init__(self, max_node_size: 10000):
    super(DataflowGraphBatcher, self).__init__(
      features=graph_tuple_builder.Feature(
        "node", "data_flow_root_node", "int64"
      ),
      labels=graph_tuple_builder.Feature("node", "data_flow_value", "int64"),
    )
    self.max_node_size = max_node_size

  def Build(self):
    return BatchData(
      graph_ids=[1], model_data=super(DataflowGraphBatcher, self).Build(),
    )

  def BuildBatches(
    self,
    graph_pairs: Iterable[
      Tuple[
        program_graph_pb2.ProgramGraph,
        program_graph_features_pb2.ProgramGraphFeatures,
      ]
    ],
  ):
    start = time.time()
    node_size = 0
    for graph, features in graph_pairs:
      if node_size + len(graph.node) > self.max_node_size:
        start = time.time()
        yield self.Build()
        node_size = 0
      node_size += len(graph.node)
      self.AddProgramGraphFeatures(graph, features)
    if node_size:
      yield self.Build()


def Main():
  """Run the model with the requested flags actions.

  Args:
    model_class: The model to run.

  Returns:
    A DataFrame of k-fold results, or a single series of results.
  """
  FLAGS.batch_results_averaging_method = "binary"

  path = pathlib.Path(FLAGS.path)
  (path / "ggnn").mkdir(exist_ok=True)

  epoch_count = 300
  model = Ggnn(
    test_only=False,
    node_y_dimensionality=2,
    graph_y_dimensionality=0,
    graph_x_dimensionality=0,
  )

  data_loader = DataflowGraphLoader(path, "reachability")

  for epoch_num in range(1, epoch_count + 1):
    start_time = time.time()
    epoch_results = []
    for epoch_type in [epoch_pb2.TRAIN, epoch_pb2.VAL]:
      batch_builder = DataflowGraphBatcher(max_node_size=10000)

      rolling_results = RollingResults()
      graphs = ppar.ThreadedIterator(
        data_loader.LoadGraphs(epoch_type), max_queue_size=5
      )
      batches = ppar.ThreadedIterator(
        batch_builder.BuildBatches(graphs), max_queue_size=5
      )
      for batch_data in batches:
        batch_results = model.RunBatch(epoch_type, batch_data)
        app.Log(1, "batch %s", batch_results)
        rolling_results.Update(batch_data, batch_results, weight=None)
        app.Log(1, "%s", batch_results.predictions)
      epoch_results.append(rolling_results.ToEpochResults())
      print(epoch_results[-1])

    epoch = epoch_pb2.Epoch(
      walltime_seconds=time.time() - start_time,
      epoch_num=epoch_num,
      train_results=epoch_results[0],
      val_results=epoch_results[1],
    )
    app.Log(1, "epoch %d\n%s", epoch_num, epoch)


if __name__ == "__main__":
  app.Run(Main)
