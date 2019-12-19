# Copyright 2019 the ProGraML authors.
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
"""This modules defines an LSTM model for graph-level classification."""
import enum
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Optional

import numpy as np
import tensorflow as tf

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.ir import ir_database
from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models.lstm import lstm_base
from deeplearning.ml4pl.models.lstm import lstm_utils as utils
from deeplearning.ml4pl.seq import graph2seq
from deeplearning.ml4pl.seq import ir2seq
from labm8.py import app
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


app.DEFINE_enum(
  "ir2seq", Ir2SeqType, Ir2SeqType.LLVM, "The type of ir2seq encoder to use."
)
app.DEFINE_float(
  "lang_model_loss_weight", 0.2, "Weight for language model auxiliary loss."
)
app.DEFINE_integer(
  "padded_sequence_length",
  25000,
  "The padded/truncated length of encoded text sequences.",
)


class GraphLstmBatch(NamedTuple):
  """The data for an LSTM batch."""

  # Shape (batch_size, padded_sequence_length, 1), dtype np.int32
  encoded_sequences: np.array
  # Shape (batch_size, graph_x_dimensionality), dtype np.int64
  graph_x: np.array
  # Shape (batch_size, graph_y_dimensionality), dtype np.float32
  graph_y: np.array

  @property
  def targets(self) -> np.array:
    """Return the targets for predictions.
    Shape (batch_size, graph_y_dimensionality)."""
    return self.graph_y

  @property
  def x(self) -> List[np.array]:
    """Return the model 'x' inputs."""
    return [self.encoded_sequences, self.graph_x]

  @property
  def y(self) -> List[np.array]:
    """Return the model 'y' inputs."""
    return [self.graph_y, self.graph_y]

  @staticmethod
  def GetPredictions(
    model_output, ctx: progress.ProgressContext = progress.NullContext
  ) -> np.array:
    """Reshape the model outputs to an array of predictions of same shape as
    targets."""
    del ctx  # Unused.
    return model_output[0]


class GraphLstm(lstm_base.LstmBase):
  """LSTM Model.

  This is an implementation of the DeepTune model described in:

      ﻿Cummins, C., Petoumenos, P., Wang, Z., & Leather, H. (2017).
      End-to-end Deep Learning of Optimization Heuristics. In PACT. IEEE.
  """

  def __init__(
    self,
    *args,
    ir_db: Optional[ir_database.Database] = None,
    ir2seq_encoder: Optional[ir2seq.EncoderBase] = None,
    **kwargs,
  ):
    if not ir_db and not FLAGS.ir_db:
      raise app.UsageError("--ir_db is required")
    ir_db = ir_db or FLAGS.ir_db()
    self._ir2seq_encoder = ir2seq_encoder or FLAGS.ir2seq().ToEncoder(ir_db)
    super(GraphLstm, self).__init__(*args, **kwargs)

  def CreateKerasModel(self) -> tf.compat.v1.keras.Model:
    """Construct the tensorflow computation graph."""
    sequence_input = tf.compat.v1.keras.layers.Input(
      shape=(self.padded_sequence_length,), dtype="int32", name="sequence_in",
    )
    graph_x_input = tf.compat.v1.keras.layers.Input(
      shape=(self.graph_db.graph_x_dimensionality,),
      dtype="float32",
      name="graph_x",
    )

    # Construct the LSTM language model which summarizes a sequence of shape
    # (padded_sequence_length, 1) to a vector of shape
    # (y_dimensionality, lang_model_hidden_size).
    lang_model = tf.compat.v1.keras.layers.Embedding(
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
    lang_model_out = tf.compat.v1.keras.layers.Dense(
      self.y_dimensionality, activation="sigmoid", name="langmodel_out",
    )(lang_model)

    # Construct the "heuristic model". Takes as inputs a concatenation of the
    # language model and auxiliary inputs, and outputs
    # a vector shape (y_dimensinoality).
    heuristic_model = tf.compat.v1.keras.layers.Concatenate()(
      [lang_model, graph_x_input]
    )
    heuristic_model = tf.compat.v1.keras.layers.BatchNormalization()(
      heuristic_model
    )
    heuristic_model = tf.compat.v1.keras.layers.Dense(
      FLAGS.heuristic_model_hidden_size, activation="relu", name="dense_1"
    )(heuristic_model)

    model_out = tf.compat.v1.keras.layers.Dense(
      self.y_dimensionality, activation="sigmoid", name="dense_2",
    )(heuristic_model)

    model = tf.compat.v1.keras.Model(
      inputs=[sequence_input, graph_x_input],
      outputs=[model_out, lang_model_out],
      name="lstm",
    )
    model.compile(
      optimizer="adam",
      metrics=["accuracy"],
      loss=["categorical_crossentropy", "categorical_crossentropy"],
      loss_weights=[1.0, FLAGS.lang_model_loss_weight],
    )

    return model

  def GetEncoder(self) -> graph2seq.EncoderBase:
    """Construct the graph encoder."""
    if not (
      self.graph_db.graph_y_dimensionality
      and self.graph_db.node_x_dimensionality == 2
      and self.graph_db.node_y_dimensionality == 0
    ):
      raise app.UsageError(
        f"Unsupported graph dimensionalities: {self.graph_db}"
      )

    return graph2seq.GraphEncoder(
      graph_db=self.graph_db, ir2seq_encoder=self._ir2seq_encoder
    )

  def MakeBatch(
    self,
    epoch_type: epoch.Type,
    graph_iterator: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Data:
    """Create a mini-batch of LSTM data."""
    del epoch_type  # Unused.

    graphs = self.GetBatchOfGraphs(graph_iterator)
    if not graphs:
      return batches.EndOfBatches()

    # Encode the graphs in the batch.
    encoded_sequences: List[np.array] = self.encoder.Encode(graphs, ctx=ctx)
    graph_x: List[np.array] = []
    graph_y: List[np.array] = []
    for graph in graphs:
      graph_x.append(graph.tuple.graph_x)
      graph_y.append(graph.tuple.graph_y)

    # Pad and truncate encoded sequences.
    encoded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
      encoded_sequences,
      maxlen=self.padded_sequence_length,
      dtype="int32",
      padding="pre",
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
