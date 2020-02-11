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
"""This module defines a generator for random graph tuples."""
from typing import Iterable
from typing import Optional

import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import test


FLAGS = test.FLAGS


def CreateRandomGraph(
  node_x_dimensionality: int = 1,
  node_y_dimensionality: int = 0,
  graph_x_dimensionality: int = 0,
  graph_y_dimensionality: int = 0,
  node_count: int = None,
) -> nx.MultiDiGraph:
  """Generate a random graph.

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
  proto = random_programl_generator.CreateRandomProto(
    node_x_dimensionality=node_x_dimensionality,
    node_y_dimensionality=node_y_dimensionality,
    graph_x_dimensionality=graph_x_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
    node_count=node_count,
  )
  return programl.ProgramGraphProtoToNetworkX(proto)
