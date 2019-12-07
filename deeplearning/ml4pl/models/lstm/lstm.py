"""This module defines an LSTM classifier."""
import enum
from typing import Any
from typing import Iterable
from typing import NamedTuple

import keras
import numpy as np

from deeplearning.ml4pl.models import batch as batches
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import epoch
from deeplearning.ml4pl.models import run
from deeplearning.ml4pl.models.lstm import lstm_utils as utils
from deeplearning.ml4pl.seq import graph2seq
from deeplearning.ml4pl.seq import ir2seq
from labm8.py import app
from labm8.py import progress

FLAGS = app.FLAGS


class EncoderType(enum.Enum):
  OPENCL = 1
  LLVM = 2
  INST2VEC = 3

  def ToEncoder(self) -> ir2seq.OpenClEncoder:
    if self == EncoderType.OPENCL:
      return ir2seq.OpenClEncoder()
    elif self == EncoderType.LLVM:
      return ir2seq.BytecodeEncoder()
    elif self == EncoderType.INST2VEC:
      return ir2seq.Inst2VecEncoder()
    else:
      raise NotImplementedError("unreachable")


app.DEFINE_enum("encoder", EncoderType, EncoderType.LLVM, "The encoder to use.")
app.DEFINE_integer(
  "hidden_size", 200, "The size of hidden layer(s) in the LSTM baselines."
)
app.DEFINE_integer(
  "dense_hidden_size", 32, "The size of the dense output layer."
)
app.DEFINE_float(
  "lang_model_loss_weight", 0.2, "Weight for language model auxiliary loss."
)


class LstmBatch(NamedTuple):
  """."""

  # Shape (graph_count, padded_sequence_length, vocabulary_size), dtype np.int32
  encoded_sequences: np.array
  # Shape (graph_count, graph_y_dimensionality), dtype np.int32
  graph_x: np.array
  # Shape (graph_count, graph_y_dimensionality), dtype np.float32
  graph_y: np.array


class Lstm(classifier_base.ClassifierBase):
  """LSTM model for graph classification."""

  def __init__(self, *args, **kwargs):
    super(Lstm, self).__init__(*args, **kwargs)

    utils.SetAllowedGrowthOnKerasSession()

    # The encoder which performs translation from graphs to encoded sequences.
    self.encoder = graph2seq.GraphToEncodedSequence(
      self.batcher.db, FLAGS.encoder().ToEncoder()
    )

    input_layer = keras.Input(
      shape=(self.encoder.bytecode_encoder.max_sequence_length,),
      dtype="int32",
      name="model_in",
    )

    x = keras.layers.Embedding(
      input_dim=self.encoder.bytecode_encoder.vocabulary_size_with_padding_token,
      input_length=self.encoder.bytecode_encoder.max_sequence_length,
      output_dim=FLAGS.hidden_size,
      name="embedding",
    )(input_layer)

    x = utils.MakeLstm(FLAGS.hidden_size, return_sequences=True, name="lstm_1")(
      x
    )

    x = utils.MakeLstm(FLAGS.hidden_size, name="lstm_2")(x)

    langmodel_out = keras.layers.Dense(
      self.stats.graph_features_dimensionality,
      activation="sigmoid",
      name="langmodel_out",
    )(x)

    # Auxiliary inputs.
    auxiliary_inputs = keras.Input(
      shape=(self.stats.graph_features_dimensionality,), name="aux_in"
    )

    # Heuristic model. Takes as inputs a concatenation of the language model
    # and auxiliary inputs, outputs 1-hot encoded device mapping.
    x = keras.layers.Concatenate()([x, auxiliary_inputs])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(
      FLAGS.dense_hidden_size, activation="relu", name="heuristic_1"
    )(x)
    out = keras.layers.Dense(
      self.stats.graph_labels_dimensionality,
      activation="sigmoid",
      name="heuristic_2",
    )(x)

    self.model = keras.Model(
      inputs=[input_layer, auxiliary_inputs], outputs=[out, langmodel_out]
    )
    self.model.compile(
      optimizer="adam",
      metrics=["accuracy"],
      loss=["categorical_crossentropy", "categorical_crossentropy"],
      loss_weights=[1.0, FLAGS.lang_model_loss_weight],
    )

  def MakeBatch(
    self,
    graphs: Iterable[graph_tuple_database.GraphTuple],
    ctx: progress.ProgressContext = progress.NullContext,
  ) -> batches.Data:
    """Create a mini-batch by encoding, padding, and concatenating sequences.
    """
    batch_size = 0
    encoded_sequences = []
    graph_x = []
    graph_y = []

    # TODO:
    while batch_size < FLAGS.encoded_batch_size:
      # Peel off some graph_ids to encode.
      graph_ids = []

      # TODO:
      encoded_sequences = self.encoder.Encode(graph_ids)
      break

    return LstmBatch(
      encoded_sequences=np.vstack(encoded_sequences),
      graph_x=np.vstack(graph_x),
      graph_y=np.vstack(graph_y),
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
    batch_data: LstmBatch = batch.data

    x = [batch_data.encoded_sequences, batch_data.graph_x]
    y = [batch_data.graph_y, batch_data.graph_y]

    if epoch_type == epoch.Type.TRAIN:
      loss, *_ = self.model.train_on_batch(x, y)

    predictions = self.model.predict_on_batch(x)

    return batches.Results.Create(
      targets=batch_data.graph_y, predictions=predictions
    )

  def ModelDataToSave(self):
    raise NotImplementedError("github.com/ChrisCummins/ProGraML/issues/24")

  def LoadModelData(self, data_to_load: Any):
    raise NotImplementedError("github.com/ChrisCummins/ProGraML/issues/24")


def main():
  """Main entry point."""
  run.Run(Lstm)


if __name__ == "__main__":
  app.Run(main)
