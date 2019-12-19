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
"""This module defines a generator for random graph tuples."""
from typing import Iterable
from typing import Optional

from deeplearning.ml4pl.graphs.labelled import graph_tuple
from deeplearning.ml4pl.testing import random_networkx_generator
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import test


FLAGS = test.FLAGS


def CreateRandomGraphTuple(
  disjoint_graph_count: int = 1,
  node_x_dimensionality: int = 1,
  node_y_dimensionality: int = 0,
  graph_x_dimensionality: int = 0,
  graph_y_dimensionality: int = 0,
  node_count: int = None,
) -> graph_tuple.GraphTuple:
  """Generate a random graph tuple.

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
  graphs = [
    random_networkx_generator.CreateRandomGraph(
      node_x_dimensionality=node_x_dimensionality,
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=graph_x_dimensionality,
      graph_y_dimensionality=graph_y_dimensionality,
      node_count=node_count,
    )
    for _ in range(disjoint_graph_count)
  ]

  graph_tuples = [
    graph_tuple.GraphTuple.CreateFromNetworkX(graph) for graph in graphs
  ]
  if len(graph_tuples) > 1:
    return graph_tuple.GraphTuple.FromGraphTuples(graph_tuples)
  else:
    return graph_tuples[0]


def EnumerateTestSet(
  n: Optional[int] = None,
) -> Iterable[graph_tuple.GraphTuple]:
  """Enumerate a test set of "real" graph tuples."""
  for graph in random_programl_generator.EnumerateTestSet(n=n):
    yield graph_tuple.GraphTuple.CreateFromProgramGraph(graph)
