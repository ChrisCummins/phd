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
"""Graph loader benchmark."""
import contextlib
import os
import pathlib
import tempfile

from tqdm import tqdm

from labm8.py import app
from labm8.py import prof
from programl.ml.model.ggnn.ggnn_batch_builder import GgnnModelBatchBuilder
from programl.proto import epoch_pb2
from programl.task.dataflow.graph_loader import DataflowGraphLoader
from programl.test.py.plugins import llvm_program_graph
from programl.test.py.plugins import llvm_reachability_features


app.DEFINE_integer("graph_count", None, "The number of graphs to load")
app.DEFINE_integer("batch_size", 40000, "The size of batches")
FLAGS = app.FLAGS


@contextlib.contextmanager
def data_directory() -> pathlib.Path:
  """Create a dataset directory."""
  with tempfile.TemporaryDirectory() as d:
    d = pathlib.Path(d)
    (d / "labels").mkdir()
    os.symlink(llvm_program_graph.LLVM_IR_GRAPHS, d / "graphs")
    os.symlink(llvm_program_graph.LLVM_IR_GRAPHS, d / "train")
    os.symlink(llvm_program_graph.LLVM_IR_GRAPHS, d / "val")
    os.symlink(llvm_program_graph.LLVM_IR_GRAPHS, d / "test")
    os.symlink(
      llvm_reachability_features.LLVM_REACHABILITY_FEATURES,
      d / "labels" / "reachability",
    )
    yield d


def Main():
  with data_directory() as path:
    graph_loader = DataflowGraphLoader(
      path=path,
      epoch_type=epoch_pb2.TRAIN,
      analysis="reachability",
      min_graph_count=FLAGS.graph_count,
      max_graph_count=FLAGS.graph_count,
    )

    with prof.Profile("Benchmark graph loader"):
      for _ in tqdm(graph_loader, unit=" graphs"):
        pass

    batch_builder = GgnnModelBatchBuilder(
      graph_loader=DataflowGraphLoader(
        path=path,
        epoch_type=epoch_pb2.TRAIN,
        analysis="reachability",
        min_graph_count=FLAGS.graph_count,
        max_graph_count=FLAGS.graph_count,
      ),
      vocabulary={"": 0},
      max_node_size=FLAGS.batch_size,
    )

    with prof.Profile("Benchmark batch construction"):
      for _ in tqdm(batch_builder, unit=" batches"):
        pass


if __name__ == "__main__":
  app.Run(Main)
