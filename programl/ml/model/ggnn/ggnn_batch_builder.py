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
"""Batch build for GGNN graphs."""
from typing import Dict
from typing import Iterable

import numpy as np

from labm8.py import app
from programl.graph.format.py.graph_tuple_builder import GraphTupleBuilder
from programl.ml.batch.base_batch_builder import BaseBatchBuilder
from programl.ml.batch.base_graph_loader import BaseGraphLoader
from programl.ml.batch.batch_data import BatchData
from programl.ml.model.ggnn.ggnn_batch import GgnnBatchData
from programl.proto import node_pb2
from programl.proto import program_graph_features_pb2
from programl.proto import program_graph_pb2


class GgnnModelBatchBuilder(BaseBatchBuilder):
  """The GGNN batch builder.

  Constructs a graph tuple per-batch.
  """

  def __init__(
    self,
    graph_loader: BaseGraphLoader,
    vocabulary: Dict[str, int],
    max_node_size: int = 10000,
    max_batch_count: int = None,
  ):
    super(GgnnModelBatchBuilder, self).__init__(graph_loader)
    self.vocabulary = vocabulary
    self.max_node_size = max_node_size
    self.max_batch_count = max_batch_count

    # Mutable state.
    self.builder = GraphTupleBuilder()
    self.vocab_ids = []
    self.selector_ids = []
    self.node_labels = []
    self.batch_count = 0

    # Determine the type of graph loader that we are building batches from.
    iter_type = self.graph_loader.IterableType()
    if self.graph_loader.IterableType() == (
      program_graph_pb2.ProgramGraph,
      program_graph_features_pb2.ProgramGraphFeatures,
    ):
      self.use_node_list = False
    elif self.graph_loader.IterableType() == (
      program_graph_pb2.ProgramGraph,
      program_graph_features_pb2.ProgramGraphFeatures,
      node_pb2.NodeIndexList,
    ):
      self.use_node_list = True
    else:
      raise TypeError(f"Unsupported graph reader iterable type: {iter_type}")

  def _Reset(self) -> None:
    self.vocab_ids = []
    self.selector_ids = []
    self.node_labels = []

  def _Build(self) -> BatchData:
    gt = self.builder.Build()
    batch = BatchData(
      graph_count=gt.graph_size,
      model_data=GgnnBatchData(
        graph_tuple=gt,
        vocab_ids=np.array(self.vocab_ids, dtype=np.int32),
        selector_ids=np.array(self.selector_ids, dtype=np.int32),
        node_labels=np.array(self.node_labels, dtype=np.int32),
      ),
    )
    self._Reset()
    return batch

  def __iter__(self) -> Iterable[BatchData]:
    node_size = 0
    for item in self.graph_loader:

      if self.use_node_list:
        graph, features, node_list = item
      else:
        graph, features = item
        node_list = list(range(len(graph.node)))

      if node_size + len(graph.node) > self.max_node_size:
        yield self._Build()
        node_size = 0
        self.batch_count += 1
        if self.max_batch_count and self.batch_count >= self.max_batch_count:
          app.Log(2, "Stopping after producing %d batches", self.batch_count)
          # Signal to the graph reader that we do not require any more graphs.
          self.graph_loader.Stop()
          return

      self.builder.AddProgramGraph(graph)

      # Read the graph node features.
      self.vocab_ids += [
        self.vocabulary.get(graph.node[n].text, len(self.vocabulary))
        for n in node_list
      ]
      self.selector_ids += [
        features.node_features.feature_list["data_flow_root_node"]
        .feature[n]
        .int64_list.value[0]
        for n in node_list
      ]
      self.node_labels += [
        features.node_features.feature_list["data_flow_value"]
        .feature[n]
        .int64_list.value[0]
        for n in node_list
      ]

      node_size += len(graph.node)

    if node_size:
      self.batch_count += 1
      yield self._Build()

    self._Reset()
