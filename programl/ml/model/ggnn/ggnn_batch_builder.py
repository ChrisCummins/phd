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
from programl.ml.batch.base_graph_loader import BaseGraphLoader
from programl.ml.batch.batch_data import BatchData
from programl.ml.model.ggnn.ggnn_batch import GgnnBatchData


class GgnnModelBatchBuilder(object):
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
    self.graph_loader = graph_loader
    self.vocabulary = vocabulary
    self.builder = GraphTupleBuilder()
    self.max_node_size = max_node_size
    self.max_batch_count = max_batch_count

    self.vocab_ids = []
    self.selector_ids = []
    self.node_labels = []
    self.batch_count = 0

  def Reset(self) -> None:
    self.vocab_ids = []
    self.selector_ids = []
    self.node_labels = []

  def Build(self) -> BatchData:
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
    self.Reset()
    return batch

  def BuildBatches(self) -> Iterable[BatchData]:
    node_size = 0
    for graph, features in self.graph_loader:
      if node_size + len(graph.node) > self.max_node_size:
        yield self.Build()
        self.batch_count += 1
        if self.max_batch_count and self.batch_count >= self.max_batch_count:
          app.Log(2, "Stopping after producing %d batches", self.batch_count)
          # Signal to the graph reader that we do not require any more graphs.
          self.graph_loader.Stop()
          return
      self.builder.AddProgramGraph(graph)
      self.vocab_ids += [
        self.vocabulary.get(node.text, len(self.vocabulary))
        for node in graph.node
      ]
      self.selector_ids += [
        f.int64_list.value[0]
        for f in features.node_features.feature_list[
          "data_flow_root_node"
        ].feature
      ]
      self.node_labels += [
        f.int64_list.value[0]
        for f in features.node_features.feature_list["data_flow_value"].feature
      ]
      node_size += len(graph.node)

    if node_size:
      self.batch_count += 1
      yield self.Build()

    self.Reset()
