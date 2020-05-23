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
from programl.graph.format.py import graph_serializer
from programl.ml.batch.base_batch_builder import BaseBatchBuilder
from programl.ml.batch.base_graph_loader import BaseGraphLoader
from programl.ml.batch.batch_data import BatchData
from programl.ml.model.lstm.lstm_batch import LstmBatchData
from programl.proto import program_graph_features_pb2
from programl.proto import program_graph_pb2
from third_party.py.tensorflow import tf


class DataflowLstmBatchBuilder(BaseBatchBuilder):
  """The LSTM batch builder."""

  def __init__(
    self,
    graph_loader: BaseGraphLoader,
    vocabulary: Dict[str, int],
    node_y_dimensionality: int,
    batch_size: int = 256,
    padded_sequence_length: int = 256,
    max_batch_count: int = None,
  ):
    super(DataflowLstmBatchBuilder, self).__init__(graph_loader)
    self.vocabulary = vocabulary
    self.node_y_dimensionality = node_y_dimensionality
    self.batch_size = batch_size
    self.padded_sequence_length = padded_sequence_length
    self.max_batch_count = max_batch_count

    # Mutable state.
    self.graph_count = 0
    self.graph_sizes = []
    self.vocab_ids = []
    self.selector_vectors = []
    self.targets = []
    self.node_labels = []
    self.batch_count = 0

    iter_type = self.graph_loader.IterableType()
    if iter_type != (
      program_graph_pb2.ProgramGraph,
      program_graph_features_pb2.ProgramGraphFeatures,
    ):
      raise TypeError(f"Unsupported graph reader iterable type: {iter_type}")

  def _Reset(self) -> None:
    self.graph_count = 0
    self.graph_sizes = []
    self.vocab_ids = []
    self.selector_vectors = []
    self.targets = []
    self.node_labels = []

  @property
  def padding_element(self):
    return len(self.vocabulary) + 1

  def _Build(self) -> BatchData:
    self.batch_count += 1
    batch = BatchData(
      graph_count=self.graph_count,
      model_data=LstmBatchData(
        graph_sizes=np.array(self.graph_sizes, dtype=np.int32),
        encoded_sequences=tf.compat.v1.keras.preprocessing.sequence.pad_sequences(
          self.vocab_ids,
          maxlen=self.padded_sequence_length,
          dtype="int32",
          padding="pre",
          truncating="post",
          value=self.padding_element,
        ),
        selector_vectors=tf.compat.v1.keras.preprocessing.sequence.pad_sequences(
          self.selector_vectors,
          maxlen=self.padded_sequence_length,
          dtype="int32",
          padding="pre",
          truncating="post",
          value=np.zeros(2, dtype=np.int32),
        ),
        node_labels=tf.compat.v1.keras.preprocessing.sequence.pad_sequences(
          self.node_labels,
          maxlen=self.padded_sequence_length,
          dtype="int32",
          padding="pre",
          truncating="post",
          value=np.zeros(self.node_y_dimensionality, dtype=np.int32),
        ),
        targets=self.targets,
      ),
    )
    app.Log(
      1,
      "Shapes: encoded_sequences=%s, selector_vectors=%s, node_labels=%s",
      batch.model_data.encoded_sequences.shape,
      batch.model_data.selector_vectors.shape,
      batch.model_data.node_labels.shape,
    )
    self._Reset()
    return batch

  def __iter__(self) -> Iterable[BatchData]:
    for item in self.graph_loader:
      graph, features = item

      # Get the list of graph node indices that produced the serialized encoded
      # graph representation. We use this to construct predictions for the
      # "full" graph through padding.
      node_list = graph_serializer.SerializeInstructionsInProgramGraph(
        graph, self.padded_sequence_length
      )
      self.graph_sizes.append(len(node_list))

      self.vocab_ids.append(
        [
          self.vocabulary.get(
            graph.node[n]
            .features.feature["inst2vec_preprocessed"]
            .bytes_list.value[0]
            .decode("utf-8"),
            self.vocabulary["!UNK"],
          )
          for n in node_list
        ]
      )

      selector_values = np.array(
        [
          features.node_features.feature_list["data_flow_root_node"]
          .feature[n]
          .int64_list.value[0]
          for n in node_list
        ],
        dtype=np.int32,
      )
      selector_vectors = np.zeros((selector_values.size, 2), dtype=np.int32)
      selector_vectors[np.arange(selector_values.size), selector_values] = 1
      self.selector_vectors.append(selector_vectors)

      node_labels = np.array(
        [
          features.node_features.feature_list["data_flow_value"]
          .feature[n]
          .int64_list.value[0]
          for n in node_list
        ],
        dtype=np.int32,
      )
      self.targets.append(node_labels)

      node_labels_1hot = np.zeros(
        (node_labels.size, self.node_y_dimensionality), dtype=np.int32
      )
      node_labels_1hot[np.arange(node_labels.size), node_labels] = 1
      self.node_labels.append(node_labels_1hot)
      if self.graph_count >= self.batch_size:
        yield self._Build()
        if self.max_batch_count and self.batch_count >= self.max_batch_count:
          app.Log(2, "Stopping after producing %d batches", self.batch_count)
          # Signal to the graph reader that we do not require any more graphs.
          self.graph_loader.Stop()
          return
      self.graph_count += 1

    if self.graph_count:
      yield self._Build()

    self._Reset()
