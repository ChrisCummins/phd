"""Train and evaluate a model for graph-level classification."""
import typing

import keras
import numpy as np
from keras import models
from labm8 import app

from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.lstm import graph2seq

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
#
##### End of flag declarations.


class LstmGraphClassifierModel(classifier_base.ClassifierBase):
  """LSTM model for graph classification."""

  def __init__(self, *args, **kwargs):
    super(LstmGraphClassifierModel, self).__init__(*args, **kwargs)

    # The encoder which performs translation from graphs to encoded sequences.
    self.encoder = graph2seq.GraphToSequenceEncoder(self.batcher.db)

    # Language model. It begins with an optional embedding layer, then has two
    # layers of LSTM network, returning a single vector of size
    # self.lstm_layer_size. The input to the LSTM is encoded bytecode sequences,
    # padded to the same length.
    input_layer = keras.Input(shape=(self.encoder.max_sequence_length,),
                              dtype='int32',
                              name="model_in")

    lstm_input = keras.layers.Embedding(
        input_dim=self.encoder.vocabulary_size_with_padding_token,
        input_length=self.encoder.max_sequence_length,
        output_dim=FLAGS.hidden_size,
        name="embedding")(input_layer)

    x = keras.layers.LSTM(FLAGS.hidden_size,
                          implementation=1,
                          return_sequences=True,
                          name="lstm_1")(lstm_input)
    x = keras.layers.LSTM(FLAGS.hidden_size, implementation=1, name="lstm_2")(x)
    langmodel_out = keras.layers.Dense(self.stats.graph_features_dimensionality,
                                       activation="sigmoid",
                                       name="langmodel_out")(x)

    # Auxiliary inputs.
    auxiliary_inputs = keras.Input(
        shape=(self.stats.graph_features_dimensionality,), name="aux_in")

    # Heuristic model. Takes as inputs a concatenation of the language model
    # and auxiliary inputs, outputs 1-hot encoded device mapping.
    x = keras.layers.Concatenate()([x, auxiliary_inputs])
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(FLAGS.dense_hidden_size,
                           activation="relu",
                           name="heuristic_1")(x)
    out = keras.layers.Dense(self.stats.graph_labels_dimensionality,
                             activation="sigmoid",
                             name='heuristic_2')(x)

    self.model = keras.Model(inputs=[input_layer, auxiliary_inputs],
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
    for batch in self.batcher.MakeGaphBatchIterator(epoch_type):
      graph_ids = batch.log.graph_indices
      encoded_sequences = self.encoder.GraphsToEncodedBytecodes(graph_ids)
      yield batch.log, {
          'encoded_sequences': np.vstack(encoded_sequences),
          'graph_x': np.vstack(batch.graph_x),
          'graph_y': np.vstack(batch.graph_y),
      }

  def RunMinibatch(self, log: log_database.BatchLog, batch: typing.Any
                  ) -> classifier_base.ClassifierBase.MinibatchResults:
    """Run a batch through the LSTM."""
    x = [batch['encoded_sequences'], batch.graph_x]
    y = [batch.graph_y, batch.graph_y]

    losses = []

    def _RecordLoss(epoch, data):
      """Callback to record training/prediction loss."""
      del epoch
      losses.append(data['loss'])

    callbacks = [keras.callbacks.LambdaCallback(on_epoch_end=_RecordLoss)]

    if log.group == 'train':
      self.model.fit(x,
                     y,
                     epochs=1,
                     batch_size=log.graph_count,
                     callbacks=callbacks,
                     verbose=False,
                     shuffle=False)

    log.loss = sum(losses) / max(len(losses), 1)

    # Run the same input again through the LSTM to get the raw predictions.
    # This is obviously wasteful when training, but I don't know of a way to
    # get the raw predictions from self.model.fit().
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
