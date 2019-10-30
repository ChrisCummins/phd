"""Train and evaluate a model for graph classification."""
import pickle
import typing

import keras
from keras import models
from labm8 import app
from labm8 import prof

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.lstm import bytecode2seq

FLAGS = app.FLAGS

##### Beginning of flag declarations.
#
# Some of these flags define parameters which must be equal when restoring from
# file, such as the hidden layer sizes. Other parameters may change between
# runs of the same model, such as the input data batch size. To accomodate for
# this, a ClassifierBase.GetModelFlagNames() method returns the list of flags
# which must be consistent between runs of the same model.
#
# For the sake of readability, these important model flags are saved into a
# global set classifier_base.MODEL_FLAGS here, so that the declaration of model
# flags is local to the declaration of the flag.
app.DEFINE_input_path(
    "vocabulary", None, "The path to the vocabulary, as produced by "
    "//deeplearning/ml4pl/models/lstm:derive_vocabulary")

app.DEFINE_integer("hidden_size", 200, "The size of hidden layer(s).")
classifier_base.MODEL_FLAGS.add("hidden_size")

app.DEFINE_integer("dense_hidden_size", 32, "The size of the dense ")
classifier_base.MODEL_FLAGS.add("dense_hidden_size")

app.DEFINE_integer(
    "input_sequence_len", 10000,
    "The length of encoded input sequences. Sequences shorter "
    "than this are padded. Sequences longer than this are "
    "truncated.")
classifier_base.MODEL_FLAGS.add("input_sequence_len")

app.DEFINE_float('lang_model_loss_weight', .2,
                 'Weight for language model auxiliary loss.')
classifier_base.MODEL_FLAGS.add("lang_model_loss_weight")

app.DEFINE_database('bytecode_db',
                    bytecode_database.Database,
                    None,
                    'URL of database to read bytecodes from.',
                    must_exist=True)
#
##### End of flag declarations.


class LstmGraphClassifierModel(classifier_base.ClassifierBase):
  """LSTM model for graph classification."""

  def __init__(self, *args, **kwargs):
    super(LstmGraphClassifierModel, self).__init__(*args, **kwargs)

    self.bytecode_db = FLAGS.bytecode_db()

    with open(FLAGS.vocabulary, 'rb') as f:
      data_to_load = pickle.load(f)
      self.vocabulary = data_to_load['vocab']
      self.max_encoded_length = data_to_load['max_encoded_length']

    # Language model. It begins with an optional embedding layer, then has two
    # layers of LSTM network, returning a single vector of size
    # self.lstm_layer_size.
    input_layer = keras.Input(shape=self.max_encoded_length,
                              dtype='int32',
                              name="model_in")

    embedding_dim = len(self.vocabulary) + 1
    lstm_input = keras.layers.Embedding(input_dim=embedding_dim,
                                        input_length=self.max_encoded_length,
                                        output_dim=FLAGS.hidden_size,
                                        name="embedding")(input_layer)

    x = keras.layers.LSTM(FLAGS.hidden_size,
                          implementation=1,
                          return_sequences=True,
                          name="lstm_1")(lstm_input)
    x = keras.layers.LSTM(FLAGS.hidden_size, implementation=1, name="lstm_2")(x)
    langmodel_out = keras.layers.Dense(self.stats.graph_features_dimensionality,
                                       activation="sigmoid")(x)

    # Auxiliary inputs.
    auxiliary_inputs = keras.Input(
        shape=(self.stats.graph_features_dimensionality,), name="aux_in")

    # Heuristic model. Takes as inputs a concatenation of the language model
    # and auxiliary inputs, outputs 1-hot encoded device mapping.
    x = keras.layers.Concatenate()([auxiliary_inputs, x])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(FLAGS.dense_hidden_size, activation="relu")(x)
    out = keras.layers.Dense(self.stats.graph_features_dimensionality,
                             activation="sigmoid")(x)

    self.model = keras.Model(inputs=[auxiliary_inputs, input_layer],
                             outputs=[out, langmodel_out])
    self.model.compile(
        optimizer="adam",
        metrics=['accuracy'],
        loss=["categorical_crossentropy", "categorical_crossentropy"],
        loss_weights=[1., FLAGS.lang_model_loss_weight])

  def MakeMinibatchIterator(
      self, epoch_type: str
  ) -> typing.Iterable[typing.Tuple[log_database.BatchLog, typing.Any]]:
    """Create minibatches by encoding, padding, and concatenating text
    sequences."""
    for batch in self.batcher.MakeGroupBatchIterator(epoch_type):
      yield batch['log'], None

  def RunMinibatch(self, epoch_type: str,
                   batch: typing.Any) -> classifier_base.MinibatchResults:
    """Pass"""
    graph_ids = batch['log'].graph_indices

    with prof.Profile('Load bytecodes'):
      # Build a mapping from graph ID to bytecode ID.
      with self.batcher.db.Session() as session:
        query = session.query(
            graph_database.GraphMeta.id,
            graph_database.GraphMeta.bytecode_id) \
          .filter(graph_database.GraphMeta.id.in_(graph_ids))

        graph_to_bytecode_ids = {row[0]: row[1] for row in query}

      # Load the bytecode strings in the order of the graphs.
      with self.bytecode_db.Session() as session:
        query = session.query(
            bytecode_database.LlvmBytecode.id,
            bytecode_database.LlvmBytecode.bytecode) \
          .filter(bytecode_database.LlvmBytecode.id.in_(
            graph_to_bytecode_ids.values()))

        results = sorted(query, key=lambda row: graph_ids.index(row.id))

    bytecodes = [row.bytecode for row in results]

    with prof.Profile('Encoded sequences'):
      encoded_sequences = bytecode2seq.Encode(bytecodes, self.vocabulary)

    history = self.model.fit([encoded_sequences, batch['graph_x']],
                             batch['graph_y'],
                             epochs=1,
                             batch_size=FLAGS.batch_size,
                             verbose=True,
                             shuffle=False)

    app.Log(1, "HISTORY %s", history.keys())
    # log.loss = history.

  def ModelDataToSave(self):
    model_path = self.working_dir / f'{self.run_id}_keras_model.h5'
    self.model.save(model_path)
    return {'model_path': model_path}

  def LoadModelData(self, data_to_load: typing.Any):
    model_path = data_to_load['model_path']
    models.load_model(model_path)


def main():
  """Main entry point."""
  classifier_base.Run(LstmGraphClassifierModel)


if __name__ == '__main__':
  app.Run(main)