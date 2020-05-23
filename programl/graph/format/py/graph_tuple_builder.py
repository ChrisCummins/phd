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
"""TODO."""
import numpy as np

from programl.graph.format.py.graph_tuple import GraphTuple
from programl.proto import program_graph_pb2


class GraphTupleBuilder(object):
  """TODO."""

  def __init__(self):
    self.graph_size = 0
    self.node_size = 0
    self.edge_size = 0
    self.node_sizes = []
    self.edge_sizes = []
    self._adjacencies = [[], [], []]
    self._edge_positions = [[], [], []]

    self.Clear()

  def Clear(self):
    self.graph_size = 0
    self.node_size = 0
    self.edge_size = 0
    self.node_sizes = []
    self.edge_sizes = []
    self._adjacencies = [[], [], []]
    self._edge_positions = [[], [], []]

  def Build(self) -> GraphTuple:
    graph_tuple = GraphTuple(
      adjacencies=self.adjacencies,
      edge_positions=self.edge_positions,
      node_sizes=self.node_sizes,
      edge_sizes=self.edge_sizes,
      graph_size=self.graph_size,
      node_size=self.node_size,
      edge_size=self.edge_size,
    )
    self.Clear()
    if not graph_tuple.graph_size:
      raise ValueError(f"Graph tuple contains no graphs: {graph_tuple}")
    if not graph_tuple.node_size:
      raise ValueError(f"Graph tuple contains no nodes: {graph_tuple}")
    if not graph_tuple.edge_size:
      raise ValueError(f"Graph tuple contains no edges: {graph_tuple}")
    return graph_tuple

  def AddProgramGraph(self, graph: program_graph_pb2.ProgramGraph) -> None:
    for edge in graph.edge:
      self._adjacencies[edge.flow].append(
        (edge.source + self.node_size, edge.target + self.node_size)
      )
      self._edge_positions[edge.flow].append(edge.position)

    # Update counters. We must do this after iterating over the edges since
    # we use the current node size to calculate an offset.
    self.graph_size += 1
    self.node_size += len(graph.node)
    self.edge_size += len(graph.edge)
    self.node_sizes.append(len(graph.node))
    self.edge_sizes.append(len(graph.edge))

  @property
  def adjacencies(self) -> np.array:
    return np.array(
      [
        np.array(self._adjacencies[0], dtype=np.int32).reshape((-1, 2)),
        np.array(self._adjacencies[1], dtype=np.int32).reshape((-1, 2)),
        np.array(self._adjacencies[2], dtype=np.int32).reshape((-1, 2)),
      ]
    )

  @property
  def edge_positions(self) -> np.array:
    return np.array(
      [
        np.array(self._edge_positions[0], dtype=np.int32),
        np.array(self._edge_positions[1], dtype=np.int32),
        np.array(self._edge_positions[2], dtype=np.int32),
      ]
    )
