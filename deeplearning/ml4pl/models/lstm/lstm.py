"""This module defines an LSTM classifier."""
import enum
import io
import pathlib
import tempfile
from typing import Any
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional

import keras
import numpy as np
import tensorflow as tf

from deeplearning.ml4pl.graphs.labelled import graph_database_reader
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.ir import ir_database
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import run
from deeplearning.ml4pl.models.lstm import lstm_utils as utils
from deeplearning.ml4pl.seq import graph2seq
from deeplearning.ml4pl.seq import ir2seq
from labm8.py import app
from labm8.py import humanize
from labm8.py import progress

FLAGS = app.FLAGS


class Ir2SeqType(enum.Enum):
  OPENCL = 1
  LLVM = 2
  INST2VEC = 3

  def ToEncoder(self, ir_db: ir_database.Database) -> ir2seq.EncoderBase:
    """Create the ir2seq encoder."""
    if self == Ir2SeqType.OPENCL:
      return ir2seq.OpenClEncoder(ir_db)
    elif self == Ir2SeqType.LLVM:
      return ir2seq.LlvmEncoder(ir_db)
    elif self == Ir2SeqType.INST2VEC:
      return ir2seq.Inst2VecEncoder(ir_db)
    else:
      raise NotImplementedError("unreachable")


class NodeEncoder(enum.Enum):
  STATEMENT = 1
  IDENTIFIER = 2

  def ToEncoder(
    self,
    graph_db: graph_tuple_database.Database,
    ir2seq_encoder: ir2seq.EncoderBase,
  ) -> graph2seq.EncoderBase:
    """Create the graph2seq encoder."""
    if self == NodeEncoder.STATEMENT:
      return graph2seq.StatementEncoder(
        graph_db=graph_db, ir2seq_encoder=ir2seq_encoder
      )
    elif self == NodeEncoder.IDENTIFIER:
      return graph2seq.IdentifierEncoder(
        graph_db=graph_db, ir2seq_encoder=ir2seq_encoder
      )
    else:
      raise NotImplementedError("unreachable")


app.DEFINE_enum(
  "ir2seq", Ir2SeqType, Ir2SeqType.LLVM, "The type of ir2seq encoder to use."
)
app.DEFINE_enum(
  "nodes",
  NodeEncoder,
  # No default value because the presence of this flag is used to select
  # between graph or node-level models.
  None,
  "The types of nodes segmentation to perform.",
)
app.DEFINE_integer(
  "lang_model_hidden_size",
  64,
  "The size of hidden layer(s) in the LSTM baselines.",
)
app.DEFINE_integer(
  "heuristic_model_hidden_size", 32, "The size of the dense output layer."
)
app.DEFINE_float(
  "lang_model_loss_weight", 0.2, "Weight for language model auxiliary loss."
)
app.DEFINE_integer(
  "padded_sequence_length", 25000, "The size of the dense output layer."
)
app.DEFINE_integer(
  "max_nodes_in_graph",
  25000,
  "The maximum number of segmented nodes to process per graph.",
)
app.DEFINE_integer(
  "batch_size",
  64,
  "The number of padded sequences to concatenate into a batch.",
)


class GraphLstmBatch(NamedTuple):
  """The data for an LSTM batch."""

  # Shape (batch_size, padded_sequence_length, 1), dtype np.int32
  encoded_sequences: np.array
  # Shape (batch_size, graph_x_dimensionality), dtype np.int32
  graph_x: np.array
  # Shape (batch_size, graph_y_dimensionality), dtype np.float32
  graph_y: np.array


class LstmBase(classifier_base.ClassifierBase):
  """An LSTM model for graph-level classification."""

  def __init__(
    self,
    *args,
    padded_sequence_length: Optional[int] = None,
    ir_db: Optional[ir_database.Database] = None,
    ir2seq_encoder: Optional[ir2seq.EncoderBase] = None,
    graph2seq_encoder: Optional[graph2seq.EncoderBase] = None,
    batch_size: Optional[int] = None,
    **kwargs,
  ):
    super(LstmBase, self).__init__(*args, **kwargs)

    # Reset any previous session.
    tf.keras.backend.clear_session()

    # The Tensorflow session and graph for the model.
    self.session = utils.SetAllowedGrowthOnKerasSession()
    self.graph = tf.compat.v1.get_default_graph()

    self.batch_size = batch_size or FLAGS.batch_size

    # Create the sequence encoder.
    if not ir_db and not FLAGS.ir_db:
      raise app.UsageError("--ir_db is required")
    self.ir2seq_encoder = ir2seq_encoder or FLAGS.ir2seq().ToEncoder(
      ir_db or FLAGS.ir_db()
    )

    # Create the graph encoder.
    if graph2seq_encoder:
      self.encoder = graph2seq_encoder
    elif (
      self.graph_db.graph_y_dimensionality
      and self.graph_db.node_x_dimensionality == 2
    ):
      # Graph-level classification.
      self.encoder = graph2seq.GraphEncoder(
        graph_db=self.graph_db, ir2seq_encoder=self.ir2seq_encoder
      )
    elif (
      self.graph_db.node_y_dimensionality
      and self.graph_db.node_x_dimensionality == 2
    ):
      # Node-wise classification with selector vector.
      if not FLAGS.nodes:
        raise app.UsageError("--nodes is required")
      self.encoder = FLAGS.nodes().ToEncoder(self.graph_db, self.ir2seq_encoder)
    else:
      raise TypeError("Unsupported graph dimensionalities")

    # Determine the size of padded sequences. Use the requested
    # padded_sequence_length, or the maximum encoded length if it is shorter.
    padded_sequence_length = (
      padded_sequence_length or FLAGS.padded_sequence_length
    )
    self.padded_sequence_length = min(
      padded_sequence_length, self.encoder.max_encoded_length
    )

    # Set by subclasses.
    self.model = None

  def Summary(self) -> str:
    """Get a summary"""
    buf = io.StringIO()
    self.model.summary(print_fn=lambda msg: print(msg, file=buf))
    print(
      "Using padded sequence length "
      f"{humanize.DecimalPrefix(self.padded_sequence_length, '')} from maximum "
      f"{humanize.DecimalPrefix(self.encoder.max_encoded_length, '')} "
      f"(max {(1 - (self.padded_sequence_length / self.encoder.max_encoded_length)):.3%} "
      "sequence truncation)",
      file=buf,
    )
    return buf.getvalue()

  @property
  def padded_vocabulary_size(self) -> int:
    return self.encoder.vocabulary_size + 1

  @property
  def padding_element(self) -> int:
    return self.encoder.vocabulary_size

  def GetBatchOfGraphs(
    self, graph_iterator: Iterable[graph_tuple_database.GraphTuple]
  ) -> List[graph_tuple_database.GraphTuple]:
    """Read a list of <= batch_size graphs from a graph_iterator."""
    # Peel off a batch of graphs to process.
    graphs: List[graph_tuple_database.GraphTuple] = []
    while len(graphs) < self.batch_size:
      try:
        graph = next(graph_iterator)
      except StopIteration:
        # We have run out of graphs.
        break
      graphs.append(graph)
    return graphs

  def GetModelData(self) -> Any:
    """Get the model state."""
    # According to https://keras.io/getting-started/faq/, it is not recommended
    # to pickle a Keras model. So as a workaround, I use Keras's saving
    # mechanism to store the weights, and pickle that.
    with tempfile.TemporaryDirectory(prefix="lstm_pickle_") as d:
      path = pathlib.Path(d) / "weights.h5"
      with self.graph.as_default():
        tf.compat.v1.keras.backend.set_session(self.session)
        self.model.save(path)

      with open(path, "rb") as f:
        model_data = f.read()
    return model_data

  def LoadModelData(self, data_to_load: Any) -> None:
    """Restore the model state."""
    # Load the weights from a file generated by ModelDataToSave().
    with tempfile.TemporaryDirectory(prefix="lstm_pickle_") as d:
      path = pathlib.Path(d) / "weights.h5"
      with open(path, "wb") as f:
        f.write(data_to_load)

    with self.graph.as_default():
      tf.compat.v1.keras.backend.set_session(self.session)
      self.model = tf.keras.models.load_model(path)


class GraphLstm(LstmBase):
  """LSTM Model.

  This is an implementation of the DeepTune model described in:

      ﻿Cummins, C., Petoumenos, P., Wang, Z., & Leather, H. (2017).
      End-to-end Deep Learning of Optimization Heuristics. In PACT. IEEE.
  """

  def __init__(self, *args, **kwargs):
    super(GraphLstm, self).__init__(*args, **kwargs)

    # To enable thread-safe use of a Keras model we must make sure to fix the
    # graph and session whenever we are going to use self.model.
    with self.graph.as_default():
      tf.compat.v1.keras.backend.set_session(self.session)
      self.CreateTensorFlowModel()

      # To enable thread-safe use of the Keras model we must ensure that
      # the computation graph is fully instantiated before the first call
      # to RunBatch(). Keras lazily instantiates parts of the graph (such as
      # training ops), so make sure those are created by running the training
      # loop now on a single graph.
      reader = graph_database_reader.BufferedGraphReader(self.graph_db, limit=1)
      batch = self.MakeBatch(reader)
      assert batch.graph_count == 1
      self.RunBatch(epoch.Type.TRAIN, batch)

      # Run private model methods that instantiate graph components.
      # See: https://stackoverflow.com/a/46801607
      self.model._make_predict_function()
      self.model._make_test_function()
      self.model._make_train_function()

      # Saving the graph also creates new ops, so run it now.
      with tempfile.TemporaryDirectory(prefix="ml4pl_lstm_") as d:
        self.model.save(pathlib.Path(d) / "delete_md.h5")

    # Finall we have instantiated the graph, so freeze it to mane any
    # implicit modification raise an error.
    self.graph.finalize()

  def CreateTensorFlowModel(self):
    """Construct the tensorflow computation graph."""
    sequence_input = keras.Input(
      shape=(self.padded_sequence_length,), dtype="int32", name="sequence_in",
    )
    graph_x_input = keras.Input(
      shape=(self.graph_db.graph_x_dimensionality,), name="graph_x"
    )

    # Construct the LSTM language model which summarizes a sequence of shape
    # (padded_sequence_length, 1) to a vector of shape
    # (y_dimensionality, lang_model_hidden_size).
    lang_model = keras.layers.Embedding(
      input_dim=self.padded_vocabulary_size,
      input_length=self.padded_sequence_length,
      output_dim=FLAGS.lang_model_hidden_size,
      name="embedding",
    )(sequence_input)
    lang_model = utils.LstmLayer(
      FLAGS.lang_model_hidden_size, return_sequences=True, name="lstm_1"
    )(lang_model)
    lang_model = utils.LstmLayer(FLAGS.lang_model_hidden_size, name="lstm_2")(
      lang_model
    )

    # An auxiliary output used for tuning the language model independently of
    # the graph features.
    lang_model_out = keras.layers.Dense(
      self.y_dimensionality, activation="sigmoid", name="langmodel_out",
    )(lang_model)

    # Construct the "heuristic model". Takes as inputs a concatenation of the
    # language model and auxiliary inputs, and outputs
    # a vector shape (y_dimensinoality).
    heuristic_model = keras.layers.Concatenate()([lang_model, graph_x_input])
    heuristic_model = keras.layers.BatchNormalization()(heuristic_model)
    heuristic_model = keras.layers.Dense(
      FLAGS.heuristic_model_hidden_size, activation="relu", name="dense_1"
    )(heuristic_model)

    model_out = keras.layers.Dense(
      self.y_dimensionality, activation="sigmoid", name="dense_2",
    )(heuristic_model)

    self.model = keras.Model(
      inputs=[sequence_input, graph_x_input],
      outputs=[model_out, lang_model_out],
      name="lstm",
    )
    self.model.compile(
      optimizer="adam",
      metrics=["accuracy"],
      loss=["categorical_crossentropy", "categorical_crossentropy"],
      loss_weights=[1.0, FLAGS.lang_model_loss_weight],
    )

  def MakeBatch(
    self,
    graph_iterator: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Data:
    """Create a mini-batch of LSTM data."""
    graphs = self.GetBatchOfGraphs(graph_iterator)
    if not graphs:
      return batches.Data(graph_ids=[], data=None)

    # Encode the graphs in the batch.
    encoded_sequences: List[np.array] = self.encoder.Encode(graphs, ctx=ctx)
    graph_x: List[np.array] = []
    graph_y: List[np.array] = []
    for graph in graphs:
      graph_x.append(graph.tuple.graph_x)
      graph_y.append(graph.tuple.graph_y)

    # Pad and truncate encoded sequences.
    encoded_sequences = keras.preprocessing.sequence.pad_sequences(
      encoded_sequences,
      maxlen=self.padded_sequence_length,
      dtype="int32",
      padding="post",
      truncating="post",
      value=self.padding_element,
    )

    return batches.Data(
      graph_ids=[graph.id for graph in graphs],
      data=GraphLstmBatch(
        encoded_sequences=np.vstack(encoded_sequences),
        graph_x=np.vstack(graph_x),
        graph_y=np.vstack(graph_y),
      ),
    )

  def RunBatch(
    self,
    epoch_type: epoch.Type,
    batch: batches.Data,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Results:
    """Run a batch.

    Args:
      epoch_type: The epoch type.
      batch: The batch to process.
      ctx: Logging context.

    Returns:
      Batch results.
    """
    batch_data: GraphLstmBatch = batch.data

    x = [batch_data.encoded_sequences, batch_data.graph_x]
    y = [batch_data.graph_y, batch_data.graph_y]

    # We can only get the loss on training.
    loss = None

    with self.graph.as_default():
      tf.compat.v1.keras.backend.set_session(self.session)

      if epoch_type == epoch.Type.TRAIN:
        loss, *_ = self.model.train_on_batch(x, y)

      predictions, *_ = self.model.predict_on_batch(x)

    return batches.Results.Create(
      targets=batch_data.graph_y, predictions=predictions, loss=loss
    )


class NodeLstmBatch(NamedTuple):
  """The data for an LSTM batch."""

  # Shape (batch_size, padded_sequence_length, 1), dtype np.int32
  encoded_sequences: np.array
  # Shape (batch_size, padded_sequence_length, 1), dtype np.int32
  segment_ids: np.array
  # Shape (batch_size, max_nodes_in_graph, 2), dtype np.int32
  selector_vectors: np.array
  # Shape (batch_size, max_nodes_in_graph, node_y_dimensionality),
  # dtype np.float32
  node_y: np.array


class NodeLstm(LstmBase):
  """An extension of the LSTM model to support node-level classification."""

  def __init__(self, *args, max_nodes_in_graph: Optional[int] = None, **kwargs):
    super(NodeLstm, self).__init__(*args, **kwargs)

    if isinstance(self.ir2seq_encoder, ir2seq.OpenClEncoder):
      raise TypeError(
        "OpenCL encoder is not supported for node-level " "classification"
      )

    self.max_nodes_in_graph = min(
      self.padded_sequence_length,
      max_nodes_in_graph or FLAGS.max_nodes_in_graph,
    )

    sequence_input = keras.Input(
      batch_shape=(self.batch_size, self.padded_sequence_length,),
      dtype="int32",
      name="sequence_in",
    )
    segment_ids = keras.Input(
      batch_shape=(self.batch_size, self.padded_sequence_length,),
      dtype="int32",
      name="segment_ids",
    )
    selector_vector = keras.Input(
      batch_shape=(self.batch_size, None, 2), name="selector_vector"
    )

    # Embed the sequence inputs and sum the embeddings by nodes.
    embedded_inputs = keras.layers.Embedding(
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
      max_output_sequence_length=self.max_nodes_in_graph,
    )
    # Because padded values in segment_ids have value, the segment sum emits a
    # tensor of shape [B, max_nodes_in_graph + 1, 64] IFF something was padded
    # and [B, max_nodes_in_graph, 64] otherwise. We want to discard the + 1
    # guy because that is just summed padded tokens anyway.
    segmented_input = utils.SliceToSizeLayer(
      segmented_input=segmented_input, selector_vector=selector_vector
    )
    lang_model_input = keras.layers.Concatenate(axis=2)(
      [segmented_input, selector_vector]
    )

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

    node_out = keras.layers.Dense(
      self.graph_db.node_y_dimensionality,
      activation="sigmoid",
      name="node_out",
    )(lang_model)

    self.model = keras.Model(
      inputs=[sequence_input, segment_ids, selector_vector], outputs=[node_out],
    )
    self.model.compile(
      optimizer="adam",
      metrics=["accuracy"],
      loss=["categorical_crossentropy"],
      loss_weights=[1.0],
    )

  def MakeBatch(
    self,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Data:
    """Create a mini-batch of LSTM data."""
    # Set the logging context for the encoder.
    self.encoder.ir2seq_encoder.ctx = ctx

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

    for graph, encoded in zip(graphs, self.encoder.Encode(graphs, ctx=ctx)):
      encoded_sequences.append(encoded.encoded_sequence)
      segment_ids.append(encoded.segment_ids)
      # Use only the 'binary selector' feature and convert to an array of
      # 1 hot binary vectors.
      node_selectors = graph.tuple.node_x[:, 1][encoded.node_mask]
      node_selector_vectors = np.zeros((node_selectors.size, 2), dtype=np.int32)
      node_selector_vectors[np.arange(node_selectors.size), node_selectors] = 1

      selector_vectors.append(node_selector_vectors)
      node_y.append(graph.tuple.node_y[encoded.node_mask])

    # Pad and truncate encoded sequences.
    encoded_sequences = keras.preprocessing.sequence.pad_sequences(
      encoded_sequences,
      maxlen=self.padded_sequence_length,
      dtype="int32",
      padding="post",
      truncating="post",
      value=self.padding_element,
    )

    # Determine an out-of-range segment ID to pad the segment IDs to.
    segment_id_padding_element = max(max(s) for s in segment_ids) + 1

    segment_ids = keras.preprocessing.sequence.pad_sequences(
      segment_ids,
      maxlen=self.padded_sequence_length,
      value=segment_id_padding_element,
    )

    selector_vectors = keras.preprocessing.sequence.pad_sequences(
      selector_vectors,
      maxlen=self.max_nodes_in_graph,
      value=np.array((0, 0), dtype=np.int32),
    )

    node_y = keras.preprocessing.sequence.pad_sequences(
      node_y,
      maxlen=self.max_nodes_in_graph,
      value=np.zeros(self.graph_db.node_y_dimensionality, dtype=np.int32),
    )

    return batches.Data(
      graph_ids=[graph.id for graph in graphs],
      data=NodeLstmBatch(
        encoded_sequences=np.vstack(encoded_sequences),
        segment_ids=np.vstack(segment_ids),
        selector_vectors=np.vstack(selector_vectors),
        node_y=np.vstack(node_y),
      ),
    )

  def RunBatch(
    self,
    epoch_type: epoch.Type,
    batch: batches.Data,
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Results:
    """Run a batch.

    Args:
      epoch_type: The epoch type.
      batch: The batch to process.
      ctx: Logging context.

    Returns:
      Batch results.
    """
    batch_data: NodeLstmBatch = batch.data

    x = [
      batch_data.encoded_sequences,
      batch_data.segment_ids,
      batch_data.selector_vectors,
    ]
    y = [batch_data.node_y]

    # We can only get the loss on training.
    loss = None

    if epoch_type == epoch.Type.TRAIN:
      loss, lang_model_loss = self.model.train_on_batch(x, y)

    predictions = self.model.predict_on_batch(x)

    # TODO(github.com/ChrisCummins/ProGraML/issues/24): Reshape node_y and
    # to predictions to match the actual graph shapes.

    return batches.Results.Create(
      targets=batch_data.node_y, predictions=predictions, loss=loss
    )


def main():
  """Main entry point."""
  if FLAGS.nodes:
    run.Run(NodeLstm)
  else:
    run.Run(GraphLstm)


if __name__ == "__main__":
  app.Run(main)
