"""This module defines an LSTM for node-level classification."""
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional

import numpy as np
import tensorflow as tf

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models.lstm import lstm_base
from deeplearning.ml4pl.models.lstm import lstm_utils as utils
from deeplearning.ml4pl.seq import graph2seq
from labm8.py import app
from labm8.py import progress


FLAGS = app.FLAGS

app.DEFINE_integer(
  "padded_nodes_sequence_length",
  5000,
  "For node-level models, the padded/truncated length of encoded node "
  "sequences.",
)


class NodeLstmBatch(NamedTuple):
  """The data for an LSTM batch."""

  # Shape (batch_size, padded_sequence_length, 1), dtype np.int32
  encoded_sequences: np.array
  # Shape (batch_size, padded_sequence_length, 1), dtype np.int32
  segment_ids: np.array
  # Shape (batch_size, padded_nodes_sequence_length, 2), dtype np.int32
  selector_vectors: np.array
  # Shape (batch_size, padded_nodes_sequence_length, node_y_dimensionality),
  # dtype np.float32
  node_y: np.array

  # A list of indices into the graph nodes for the nodes that are encoded.
  # Shape (batch_size, ?),
  node_indices: List[np.array]

  @property
  def x(self):
    return [self.encoded_sequences, self.segment_ids, self.selector_vectors]

  @property
  def y(self):
    return [self.node_y]


class NodeLstm(lstm_base.LstmBase):
  """An LSTM model for node-level classification."""

  def __init__(
    self,
    *args,
    padded_nodes_sequence_length: Optional[int] = None,
    proto_db: Optional[unlabelled_graph_database.Database] = None,
    **kwargs,
  ):
    if not proto_db and not FLAGS.proto_db:
      raise app.UsageError("--proto_db is required for node level models")
    self._proto_db = proto_db or FLAGS.proto_db()

    # Determine the maximum node sequence length.
    self._padded_nodes_sequence_length = padded_nodes_sequence_length

    super(NodeLstm, self).__init__(*args, **kwargs)

  @property
  def padded_node_sequence_length(self) -> int:
    """Get the length of padded node sequences."""
    return min(
      self.padded_sequence_length,
      self._padded_nodes_sequence_length or FLAGS.padded_nodes_sequence_length,
    )

  @property
  def warm_up_batch_size(self) -> int:
    """This model has fixed sized batches, so the warm-up iteration requires
    a full batch_size number of graphs."""
    return self.batch_size

  def CreateKerasModel(self) -> tf.compat.v1.keras.Model:
    """Construct the tensorflow computation graph."""
    sequence_input = tf.compat.v1.keras.layers.Input(
      batch_shape=(self.batch_size, self.padded_sequence_length,),
      dtype="int32",
      name="sequence_in",
    )
    segment_ids = tf.compat.v1.keras.layers.Input(
      batch_shape=(self.batch_size, self.padded_sequence_length,),
      dtype="int32",
      name="segment_ids",
    )
    selector_vector = tf.compat.v1.keras.layers.Input(
      batch_shape=(self.batch_size, None, 2),
      dtype="float32",
      name="selector_vector",
    )

    # Embed the sequence inputs and sum the embeddings by nodes.
    embedded_inputs = tf.compat.v1.keras.layers.Embedding(
      input_dim=self.padded_vocabulary_size,
      input_length=self.padded_sequence_length,
      output_dim=FLAGS.lang_model_hidden_size,
      name="embedding",
    )(sequence_input)

    segmented_input = utils.SegmentSumLayer(
      encoded_sequences=embedded_inputs,
      segment_ids=segment_ids,
      batch_size=self.batch_size,
      max_sequence_length=self.padded_sequence_length,
      max_output_sequence_length=self.graph_db.node_count_max,
    )
    # Because padded values in segment_ids have value, the segment sum emits a
    # tensor of shape [B, padded_nodes_sequence_length + 1, 64] IFF something
    # was padded and [B, padded_nodes_sequence_length, 64] otherwise. We want
    # to discard the + 1 guy because that is just summed padded tokens anyway.
    segmented_input = utils.SliceToSizeLayer(
      segmented_input=segmented_input, selector_vector=selector_vector
    )
    lang_model_input = tf.compat.v1.keras.layers.Concatenate(
      axis=2, name="segmented_inputs_and_selector_vectors"
    )([segmented_input, selector_vector],)

    # Make the language model.
    lang_model = utils.LstmLayer(
      FLAGS.lang_model_hidden_size, return_sequences=True, name="lstm_1"
    )(lang_model_input)
    lang_model = utils.LstmLayer(
      FLAGS.lang_model_hidden_size,
      return_sequences=True,
      return_state=False,
      name="lstm_2",
    )(lang_model)

    node_out = tf.compat.v1.keras.layers.Dense(
      self.graph_db.node_y_dimensionality,
      activation="sigmoid",
      name="node_out",
    )(lang_model)

    model = tf.compat.v1.keras.Model(
      inputs=[sequence_input, segment_ids, selector_vector], outputs=[node_out],
    )
    model.compile(
      optimizer="adam",
      metrics=["accuracy"],
      loss=["categorical_crossentropy"],
      loss_weights=[1.0],
    )

    return model

  def GetEncoder(self) -> graph2seq.EncoderBase:
    """Construct the graph encoder."""
    if not (
      self.graph_db.node_y_dimensionality
      and self.graph_db.node_x_dimensionality == 2
      and self.graph_db.graph_y_dimensionality == 0
    ):
      raise app.UsageError(
        f"Unsupported graph dimensionalities: {self.graph_db}"
      )
    return graph2seq.StatementEncoder(
      graph_db=self.graph_db, proto_db=self._proto_db
    )

  def MakeBatch(
    self,
    epoch_type: epoch.Type,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Data:
    """Create a mini-batch of LSTM data."""
    del epoch_type  # Unused.

    graphs = self.GetBatchOfGraphs(graphs)
    # For node classification we require all batches to be of batch_size.
    # In the future we could work around this by padding an incomplete
    # batch with arrays of zeros.
    if not graphs or len(graphs) != self.batch_size:
      return batches.Data(graph_ids=[], data=None)

    # Encode the graphs in the batch.

    # A list of arrays of shape (node_count, 1)
    encoded_sequences: List[np.array] = []
    # A list of arrays of shape (node_count, 1)
    segment_ids: List[np.array] = []
    # A list of arrays of shape (node_mask_count, 2)
    selector_vectors: List[np.array] = []
    # A list of arrays of shape (node_mask_count, node_y_dimensionality)
    node_y: List[np.array] = []
    node_indices: List[np.array] = []

    # Convert ProgramGraphSeq protos to arrays of numeric values.
    for graph, seq in zip(graphs, self.encoder.Encode(graphs, ctx=ctx)):
      # Skip empty graphs.
      if not seq.encoded:
        continue

      encoded_sequences.append(np.array(seq.encoded, dtype=np.int32))

      # Construct a list of segment IDs using the encoded node lengths,
      # e.g. for encoded node lengths [2, 3, 1], produce segment IDs:
      # [0, 0, 1, 1, 1, 2].
      out_of_range_segment = self.padded_nodes_sequence_length - 1
      segment_ids.append(
        np.concatenate(
          [
            np.ones(encoded_node_length, dtype=np.int32)
            * min(segment_id, out_of_range_segment)
            for segment_id, encoded_node_length in enumerate(
              seq.encoded_node_length
            )
          ]
        )
      )

      # Get the list of graph node indices that produced the serialized encoded
      # graph representation. We use this to construct predictions for the
      # "full" graph through padding.
      seq_node_indices = np.array(seq.node, dtype=np.int32)
      node_indices.append(seq_node_indices)

      # Sanity check that the node indices are in-range for this graph.
      assert len(graph.tuple.node_x) >= max(seq_node_indices)

      # Use only the 'binary selector' feature and convert to an array of
      # 1 hot binary vectors.
      node_selectors = graph.tuple.node_x[:, 1][seq_node_indices]
      node_selector_vectors = np.zeros((node_selectors.size, 2), dtype=np.int32)
      node_selector_vectors[np.arange(node_selectors.size), node_selectors] = 1
      selector_vectors.append(node_selector_vectors)

      # Select the node targets for only the active nodes.
      node_y.append(graph.tuple.node_y[seq_node_indices])

    # Pad and truncate encoded sequences.
    encoded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
      encoded_sequences,
      maxlen=self.padded_sequence_length,
      dtype="int32",
      padding="pre",
      truncating="post",
      value=self.padding_element,
    )

    # Determine an out-of-range segment ID to pad the segment IDs to.
    segment_id_padding_element = (
      max(max(s) if s.size else 0 for s in segment_ids) + 1
    )

    segment_ids = tf.keras.preprocessing.sequence.pad_sequences(
      segment_ids,
      maxlen=self.padded_sequence_length,
      dtype="int32",
      padding="pre",
      truncating="post",
      value=segment_id_padding_element,
    )

    padded_nodes_sequence_length = min(
      self.padded_nodes_sequence_length, max(len(s) for s in selector_vectors)
    )

    # Pad the selector vectors to the same shape as the segment IDs.)
    selector_vectors = tf.keras.preprocessing.sequence.pad_sequences(
      selector_vectors,
      maxlen=padded_nodes_sequence_length,
      dtype="int32",
      padding="pre",
      truncating="post",
      value=np.array((0, 0), dtype=np.int32),
    )

    node_y = tf.keras.preprocessing.sequence.pad_sequences(
      node_y,
      maxlen=padded_nodes_sequence_length,
      dtype="int32",
      padding="pre",
      truncating="post",
      value=np.zeros(self.graph_db.node_y_dimensionality, dtype=np.int64),
    )

    return batches.Data(
      graph_ids=[graph.id for graph in graphs],
      data=NodeLstmBatch(
        encoded_sequences=encoded_sequences,
        segment_ids=segment_ids,
        selector_vectors=selector_vectors,
        node_y=node_y,
        node_indices=node_indices,
      ),
    )

  def ReshapeTargets(self, targets):
    # Stack the manually-batched targets list.
    return np.vstack(targets)
