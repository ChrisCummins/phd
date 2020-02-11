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
"""This module defines a generator for random program graph protos.

When executed as a binary, this script generates a random program graph proto
and prints it to stdout. Example usage:

    $ bazel run //deeplearning/ml4pl/testing:random_programl_generator -- \
          --node_x_dimensionality=2
          --node_y_dimensionality=3
          --graph_x_dimensionality=2
          --graph_y_dimensionality=0
"""
import random

import numpy as np

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.py import random_graph_builder
from labm8.py import app


FLAGS = app.FLAGS

app.DEFINE_integer(
  "node_x_dimensionality", 2, "The dimensionality of node x vectors."
)
app.DEFINE_integer(
  "node_y_dimensionality", 0, "The dimensionality of node y vectors."
)
app.DEFINE_integer(
  "graph_x_dimensionality", 0, "The dimensionality of graph x vectors."
)
app.DEFINE_integer(
  "graph_y_dimensionality", 0, "The dimensionality of graph y vectors."
)
app.DEFINE_boolean(
  "with_data_flow", False, "Whether to generate data flow columns."
)
app.DEFINE_integer(
  "node_count", None, "The number of nodes in the randomly generated graph."
)


def CreateRandomProto(
  node_x_dimensionality: int = 1,
  node_y_dimensionality: int = 0,
  graph_x_dimensionality: int = 0,
  graph_y_dimensionality: int = 0,
  with_data_flow: bool = False,
  node_count: int = 0,
) -> programl_pb2.ProgramGraphProto:
  """Generate a random program graph.

  This generates a random graph which has sensible values for fields, but does
  not have meaningful semantics, e.g. there may be data flow edges between
  identifiers, etc. For speed, this generator guarantees only that:

    1. There is a 'root' node with outgoing call edges.
    2. Nodes are either statements, identifiers, or immediates.
    3. Nodes have text, preprocessed_text, and a single node_x value.
    4. Edges are either control, data, or call.
    5. Edges have positions.
    6. The graph is strongly connected.
  """
  builder = random_graph_builder.RandomGraphBuilder()
  proto = programl_pb2.ProgramGraphProto()
  proto.ParseFromString(builder.GetSerializedGraphProto(node_count))

  for node in proto.node:
    node.preprocessed_text = 0
    # Add the node features and labels.
    # Limit node feature values in range [0,1] to play nicely with models with
    # hardcoded "binary selector" embeddings.
    node.x[:] = np.random.randint(low=0, high=2, size=node_x_dimensionality)
    if node_y_dimensionality:
      node.y[:] = np.random.randint(low=0, high=100, size=node_y_dimensionality)

  if graph_x_dimensionality:
    proto.x[:] = np.random.randint(low=0, high=100, size=graph_x_dimensionality)

  if graph_y_dimensionality:
    proto.y[:] = np.random.randint(low=0, high=100, size=graph_y_dimensionality)

  if with_data_flow:
    proto.data_flow_steps = random.randint(1, 50)
    proto.data_flow_root_node = random.randint(0, node_count - 1)
    proto.data_flow_positive_node_count = random.randint(1, node_count - 1)

  return proto


def Main():
  """Main entry point"""
  print(
    CreateRandomProto(
      node_x_dimensionality=FLAGS.node_x_dimensionality,
      node_y_dimensionality=FLAGS.node_y_dimensionality,
      graph_x_dimensionality=FLAGS.graph_x_dimensionality,
      graph_y_dimensionality=FLAGS.graph_y_dimensionality,
      node_count=FLAGS.node_count,
    )
  )


if __name__ == "__main__":
  app.Run(Main)
