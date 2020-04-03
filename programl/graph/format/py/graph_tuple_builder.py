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

from programl.graph.format.py import graph_tuple_pybind
from programl.graph.format.py.graph_tuple import GraphTuple
from programl.proto import program_graph_pb2


class GraphTupleBuilder(graph_tuple_pybind.GraphTuple):
  """TODO."""

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
    return graph_tuple

  def AddProgramGraph(self, graph: program_graph_pb2.ProgramGraph) -> None:
    self._AddProgramGraph(graph.SerializeToString())

  @property
  def adjacencies(self) -> np.array:
    a = self._adjacencies
    return np.array(
      [
        np.array(a[0], dtype=np.int32),
        np.array(a[1], dtype=np.int32),
        np.array(a[2], dtype=np.int32),
      ]
    )

  @property
  def edge_positions(self) -> np.array:
    a = self._edge_positions
    return np.array(
      [
        np.array(a[0], dtype=np.int32),
        np.array(a[1], dtype=np.int32),
        np.array(a[2], dtype=np.int32),
      ]
    )
