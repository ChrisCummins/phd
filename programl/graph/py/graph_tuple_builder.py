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
from typing import NamedTuple

import numpy as np

from programl.graph.py import graph_tuple
from programl.proto import program_graph_features_pb2
from programl.proto import program_graph_pb2


class FeatureDescriptor(NamedTuple):
  location: str
  name: str
  field: str = "int64_list"
  dtype: np.dtype = np.int64


def Feature(location: str, name: str, type: str = "int64") -> FeatureDescriptor:
  return FeatureDescriptor(
    location=location,
    name=name,
    field=f"{type}_list",
    dtype={"int64": np.int64, "float": np.float}[type],
  )


class GraphTupleBuilder(object):
  def __init__(self, features: FeatureDescriptor, labels: FeatureDescriptor):
    self.features = features
    self.labels = labels
    self.graph_tuples = []

  @staticmethod
  def FlattenProgramGraph(
    graph: program_graph_pb2.ProgramGraph, feature: FeatureDescriptor
  ) -> np.array:
    if feature.location == "node":
      return np.vstack(
        np.array(
          [
            getattr(node.features.feature[feature.name], feature.field).value
            for node in graph.node
          ],
          dtype=feature.dtype,
        )
      )
    elif feature.location == "edge":
      return np.vstack(
        np.array(
          [
            getattr(edge.features.feature[feature.name], feature.field).value
            for edge in graph.edge
          ],
          dtype=feature.dtype,
        )
      )
    elif feature.location == "graph":
      return np.array(
        [
          np.array(
            getattr(graph.features.feature[feature.name], feature.field).value,
            dtype=feature.dtype,
          )
        ],
        dtype=feature.dtype,
      )

  @staticmethod
  def FlattenProgramGraphFeatures(
    graph: program_graph_features_pb2.ProgramGraphFeatures,
    feature: FeatureDescriptor,
  ) -> np.array:
    if feature.location == "node":
      return np.vstack(
        np.array(
          [
            getattr(node, feature.field).value
            for node in graph.node_features.feature_list[feature.name].feature
          ],
          dtype=feature.dtype,
        )
      )
    elif feature.location == "edge":
      return np.vstack(
        np.array(
          [
            getattr(edge, feature.field).value
            for edge in graph.edge_features.feature_list[feature.name].feature
          ],
          dtype=feature.dtype,
        )
      )
    elif feature.location == "graph":
      return np.array(
        [
          np.array(
            getattr(graph.feature[feature.name], feature.field).value,
            dtype=feature.dtype,
          )
        ],
        dtype=feature.dtype,
      )

  def _AddOne(
    self,
    graph: program_graph_pb2.ProgramGraph,
    features: np.array,
    labels: np.array,
  ):
    if not len(graph.node) or not len(graph.edge):
      raise ValueError("Graph contains no nodes or edges")
    if not features.shape[1]:
      raise ValueError(
        f"{self.features.location} feature not found: {self.features.name} ({self.features.dtype.__name__})"
      )
    if not labels.shape[1]:
      raise ValueError(
        f"{self.labels.location} feature not found: {self.labels.name} ({self.labels.dtype.__name__})"
      )

    adjacencies = [[], [], []]
    edge_positions = [[], [], []]
    for edge in graph.edge:
      adjacencies[edge.flow].append((edge.source, edge.target))
      edge_positions[edge.flow].append(edge.position)

    self.graph_tuples.append(
      graph_tuple.GraphTuple(
        adjacencies=[np.array(adj, dtype=np.int32) for adj in adjacencies],
        edge_positions=[
          np.array(edg, dtype=np.int32) for edg in edge_positions
        ],
        features=features,
        labels=labels,
        disjoint_nodes_sections=np.array([], dtype=np.int64),
        disjoint_features_sections=np.array([], dtype=np.int64),
        disjoint_labels_sections=np.array([], dtype=np.int64),
        node_count=len(graph.node),
      )
    )

  def AddProgramGraph(self, graph: program_graph_pb2.ProgramGraph):
    """Add a disjoint graph to the graph tuple.

    Args:
      graph: The graph to add.

    Returns:
      A GraphTuple instance.

    Raises:
      ValueError: If the graph contains no nodes or no edges.
    """
    features = self.FlattenProgramGraph(graph, self.features)
    labels = self.FlattenProgramGraph(graph, self.labels)
    self._AddOne(graph, features, labels)

  def AddProgramGraphFeatures(
    self,
    graph: program_graph_pb2.ProgramGraph,
    graph_features: program_graph_features_pb2.ProgramGraphFeatures,
  ):
    features = self.FlattenProgramGraphFeatures(graph_features, self.features)
    labels = self.FlattenProgramGraphFeatures(graph_features, self.labels)
    self._AddOne(graph, features, labels)

  def Build(self) -> graph_tuple.GraphTuple:
    if not len(self.graph_tuples):
      raise ValueError("No graph tuples to build")

    gt = graph_tuple.GraphTuple.Join(self.graph_tuples)
    self.graph_tuples = []
    return gt
