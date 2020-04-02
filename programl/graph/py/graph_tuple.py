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
"""The module implements conversion of graphs to tuples of arrays."""
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import numpy as np

from programl.proto import edge_pb2


class GraphTuple(NamedTuple):
  """The graph tuple: a compact representation of graph features and labels.

  The transformation of ProgramGraph protocol message to GraphTuples is lossy
  (omitting attributes such as node types and texts).
  """

  # A list of adjacency lists, one for each flow type, where an entry in an
  # adjacency list is a <src,dst> tuple of node indices.
  # Shape (edge_flow_count, edge_count, 2), dtype int32:
  adjacencies: np.array

  # A list of edge positions, one for each edge type. An edge position is an
  # integer in the range 0 <= x < edge_position_max.
  # Shape (edge_flow_count, edge_count), dtype int32:
  edge_positions: np.array

  # A flattened list of features. Each row is a feature vector.
  # Shape (?, feature_dimensionality), dtype {int64,float}:
  features: np.array

  # A flattened list of labels. Each row is a label vector.
  # Shape (?, label_dimensionality), dtype {int64,float}:
  labels: np.array

  # A list of integers which segment the nodes by graph. E.g. with a GraphTuple
  # of two distinct graphs, both with three nodes, nodes_list will be
  # [0, 0, 0, 1, 1, 1].
  # Shape (?), dtype int32
  disjoint_nodes_sections: np.array
  disjoint_features_sections: np.array
  disjoint_labels_sections: np.array

  # The total number of nodes across the disjoint graphs.
  node_count: int

  @property
  def edge_count(self) -> int:
    """Return the total number of edges of all flow types."""
    return sum(len(adjacency_list) for adjacency_list in self.adjacencies)

  @property
  def control_edge_count(self) -> int:
    return self.adjacencies[edge_pb2.Edge.CONTROL].shape[0]

  @property
  def data_edge_count(self) -> int:
    return self.adjacencies[edge_pb2.Edge.DATA].shape[0]

  @property
  def call_edge_count(self) -> int:
    return self.adjacencies[edge_pb2.Edge.CALL].shape[0]

  @property
  def edge_position_max(self) -> int:
    """Return the maximum edge position."""
    return max(
      position_list.max() if position_list.size else 0
      for position_list in self.edge_positions
    )

  @property
  def feature_dimensionality(self) -> int:
    """Return the dimensionality of node features."""
    return self.labels.shape[1]

  @property
  def labels_dimensionality(self) -> int:
    """Return the dimensionality of node labels."""
    return self.labels.shape[1]

  @classmethod
  def Join(cls, graph_tuples: List["GraphTuple"]):
    """Join a sequence of disjoint graph tuples.

    Args:
      graph_tuples: The GraphTuple instances to join.

    Returns:
       A GraphTuple instance.
    """
    adjacencies: List[List[Tuple[int, int]]] = [[], [], []]
    edge_positions: List[List[int]] = [[], [], []]
    disjoint_nodes_sections: List[int] = []
    disjoint_features_sections: List[int] = []
    disjoint_labels_sections: List[int] = []

    features: List[int] = []
    labels: List[int] = []

    node_count = 0
    disjoint_nodes_offset = 0
    disjoint_features_offset = 0
    disjoint_labels_offset = 0

    # Iterate over each graph, merging them.
    for disjoint_graph_count, graph in enumerate(graph_tuples):
      disjoint_nodes_offset += graph.node_count
      disjoint_nodes_sections.append(disjoint_nodes_offset)
      disjoint_features_offset += len(graph.features)
      disjoint_features_sections.append(disjoint_features_offset)
      disjoint_labels_offset += len(graph.labels)
      disjoint_labels_sections.append(disjoint_labels_offset)

      for edge_flow, (adjacency_list, position_list) in enumerate(
        zip(graph.adjacencies, graph.edge_positions)
      ):
        if adjacency_list.size:
          # Offset the adjacency list node indices.
          offset = np.array((node_count, node_count), dtype=np.int32)
          adjacencies[edge_flow].append(adjacency_list + offset)
          edge_positions[edge_flow].append(position_list)

      features.extend(graph.features)
      labels.extend(graph.labels)

      node_count += graph.node_count

    # Concatenate and convert lists to numpy arrays.
    for edge_flow in range(len(adjacencies)):
      if len(adjacencies[edge_flow]):
        adjacencies[edge_flow] = np.concatenate(adjacencies[edge_flow])
      else:
        adjacencies[edge_flow] = np.zeros((0, 2), dtype=np.int32)

      if len(edge_positions[edge_flow]):
        edge_positions[edge_flow] = np.concatenate(edge_positions[edge_flow])
      else:
        edge_positions[edge_flow] = np.array([], dtype=np.int32)

    return cls(
      adjacencies=np.array(adjacencies),
      edge_positions=np.array(edge_positions),
      features=np.array(features),
      labels=np.array(labels),
      disjoint_nodes_sections=np.array(
        disjoint_nodes_sections[:-1], dtype=np.int64
      ),
      disjoint_features_sections=np.array(
        disjoint_features_sections[:-1], dtype=np.int64
      ),
      disjoint_labels_sections=np.array(
        disjoint_labels_sections[:-1], dtype=np.int64
      ),
      node_count=node_count,
    )
