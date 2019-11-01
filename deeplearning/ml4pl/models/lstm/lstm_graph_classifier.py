"""Train and evaluate a model for graph classification."""
import pickle
import typing

import keras
import numpy as np
from keras import models

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.lstm import bytecode2seq
from labm8 import app
from labm8 import prof

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

app.DEFINE_database(
    'bytecode_db',
    bytecode_database.Database,
    None,
    'URL of database to read bytecodes from.',
    must_exist=True)

app.DEFINE_integer(
    'max_encoded_length', None,
    'Override the max_encoded_length value loaded from the vocabulary.')

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

    if FLAGS.max_encoded_length:
      self.max_encoded_length = FLAGS.max_encoded_length

    app.Log(1, 'Using %s-element vocabulary with sequence length %s',
            len(self.vocabulary), self.max_encoded_length)

    # Language model. It begins with an optional embedding layer, then has two
    # layers of LSTM network, returning a single vector of size
    # self.lstm_layer_size.
    input_layer = keras.Input(
        shape=(self.max_encoded_length,), dtype='int32', name="model_in")

    self.pad_val = len(self.vocabulary)
    assert self.pad_val not in self.vocabulary
    embedding_dim = len(self.vocabulary) + 1
    lstm_input = keras.layers.Embedding(
        input_dim=embedding_dim,
        input_length=self.max_encoded_length,
        output_dim=FLAGS.hidden_size,
        name="embedding")(input_layer)

    x = keras.layers.LSTM(
        FLAGS.hidden_size,
        implementation=1,
        return_sequences=True,
        name="lstm_1")(lstm_input)
    x = keras.layers.LSTM(FLAGS.hidden_size, implementation=1, name="lstm_2")(x)
    langmodel_out = keras.layers.Dense(
        self.stats.graph_features_dimensionality,
        activation="sigmoid",
        name="langmodel_out")(x)

    # Auxiliary inputs.
    auxiliary_inputs = keras.Input(
        shape=(self.stats.graph_features_dimensionality,), name="aux_in")

    # Heuristic model. Takes as inputs a concatenation of the language model
    # and auxiliary inputs, outputs 1-hot encoded device mapping.
    x = keras.layers.Concatenate()([x, auxiliary_inputs])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(
        FLAGS.dense_hidden_size, activation="relu", name="heuristic_1")(x)
    out = keras.layers.Dense(
        self.stats.graph_labels_dimensionality,
        activation="sigmoid",
        name='heuristic_2')(x)

    self.model = keras.Model(
        inputs=[input_layer, auxiliary_inputs], outputs=[out, langmodel_out])
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
    for batch in self.batcher.MakeGaphBatchIterator(epoch_type):
      graph_ids = batch.log.graph_indices

      with prof.Profile("Loaded and encoded bytecodes"):
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

          bytecode_id_to_string = {row[0]: row[1] for row in query}

        # Encode the bytecodes.
        encoded_bytecodes, vocab_out = bytecode2seq.Encode(
            bytecode_id_to_string.values(), self.vocabulary)
        if len(vocab_out) != len(self.vocabulary):
          raise ValueError("Encoded vocabulary has different size "
                           f"({len(vocab_out)}) than the input "
                           f"({len(self.vocabulary)})")
        bytecode_id_to_encoded = {
            id_: encoded for id_, encoded in zip(bytecode_id_to_string.keys(),
                                                 encoded_bytecodes)
        }

        encoded_sequences = [
            bytecode_id_to_encoded[graph_to_bytecode_ids[i]] for i in graph_ids
        ]

        one_hot_sequences = np.array(
            keras.preprocessing.sequence.pad_sequences(
                encoded_sequences,
                maxlen=self.max_encoded_length,
                value=self.pad_val))

      yield batch.log, {
          'sequence_1hot': np.vstack(one_hot_sequences),
          'graph_x': np.vstack(batch.graph_x),
          'graph_y': np.vstack(batch.graph_y),
      }

  def RunMinibatch(self, log: log_database.BatchLog, batch: typing.Any
                  ) -> classifier_base.ClassifierBase.MinibatchResults:
    """Pass"""
    x = [batch['sequence_1hot'], batch.graph_x]
    y = [batch.graph_y, batch.graph_y]

    losses = []

    def _RecordLoss(epoch, data):
      """Callback to record training/prediction loss."""
      del epoch
      losses.append(data['loss'])

    callbacks = [keras.callbacks.LambdaCallback(on_epoch_end=_RecordLoss)]

    if log.group == 'train':
      self.model.fit(
          x,
          y,
          epochs=1,
          batch_size=log.graph_count,
          callbacks=callbacks,
          verbose=False,
          shuffle=False)
    else:
      self.model.evaluate(
          x, y, batch_size=log.graph_count, callbacks=callbacks, verbose=False)

    log.loss = sum(losses) / max(len(losses), 1)

    pred_y = self.model.predict(x)

    return batch.graph_y, pred_y[0]

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
