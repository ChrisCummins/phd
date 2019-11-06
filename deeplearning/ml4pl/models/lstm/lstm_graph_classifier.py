"""Train and evaluate a model for graph-level classification."""
import typing

import keras
import numpy as np
import tensorflow as tf
from keras import models
from labm8 import app

from deeplearning.ml4pl.graphs.labelled.graph_tuple import graph_batcher
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.lstm import graph2seq
from deeplearning.ml4pl.models import base_utils

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

app.DEFINE_boolean(
    'node_wise_model', True,
    "hacky flag to activate node-wise classification instead of graph level classification"
)
classifier_base.MODEL_FLAGS.add("node_wise_model")
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
    # self.lstm_layer_size.
    input_layer = keras.Input(shape=(self.encoder.max_sequence_length,),
                              dtype='int32',
                              name="model_in")
    input_segments = keras.Input(shape=(self.encoder.max_sequence_length,),
                                 dtype='int32',
                                 name="model_in_segments")

    lstm_input = keras.layers.Embedding(
        input_dim=self.encoder.vocabulary_size_with_padding_token,
        input_length=self.encoder.max_sequence_length,
        output_dim=FLAGS.hidden_size,
        name="embedding")(input_layer, input_segments)

    if FLAGS.node_wise_model:
      x = keras.layers.Lambda(
          lambda inputs, indices: tf.math.unsorted_segment_sum(inputs, indices),
          name='segment_sum')(lstm_input, input_segments)

    x = keras.layers.CuDNNLSTM(FLAGS.hidden_size,
                               return_sequences=True,
                               name="lstm_1")(x)
    if FLAGS.node_wise_model:
      x = keras.layers.CuDNNLSTM(FLAGS.hidden_size,
                                 name="lstm_2",
                                 return_sequences=True,
                                 return_state=False)(x)
      langmodel_out = keras.layers.Dense(self.stats.node_labels_dimensionality,
                                         activation="sigmoid",
                                         name="langmodel_out")(x)
      # no graph level features for node classification.
      out = langmodel_out
      self.model = keras.Model(inputs=[input_layer, input_segments],
                               outputs=[out])
      self.model.compile(optimizer="adam",
                         metrics=['accuracy'],
                         loss=["categorical_crossentropy"],
                         loss_weights=[1.0])
    else:
      x = keras.layers.CuDNNLSTM(FLAGS.hidden_size, name="lstm_2")(x)

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
      self, epoch_type: str, group: str
  ) -> typing.Iterable[typing.Tuple[log_database.BatchLogMeta, typing.Any]]:
    """Create minibatches by encoding, padding, and concatenating text
    sequences."""
    options = graph_batcher.GraphBatchOptions(max_nodes=FLAGS.batch_size,
                                              group=group)
    max_instance_count = (
        FLAGS.max_train_per_epoch if epoch_type == 'train' else
        FLAGS.max_val_per_epoch if epoch_type == 'val' else None)
    for batch in self.batcher.MakeGraphBatchIterator(options,
                                                     max_instance_count):
      graph_ids = batch.log.graph_indices
      encoded_sequences = self.encoder.GraphsToEncodedBytecodes(graph_ids)
      yield batch.log, {
          'encoded_sequences': np.vstack(encoded_sequences),
          'graph_x': np.vstack(batch.graph_x),
          'graph_y': np.vstack(batch.graph_y),
      }

  def RunMinibatch(self, log: log_database.BatchLogMeta, batch: typing.Any
                  ) -> classifier_base.ClassifierBase.MinibatchResults:
    """Run a batch through the LSTM."""
    if FLAGS.node_wise_model:
      x = [batch['encoded_sequences'][0],
           batch['encoded_sequences'][1]]  # for clarity
      y = [batch['node_y']]
    else:
      x = [batch['encoded_sequences'], batch['graph_x']]
      y = [batch['graph_y'], batch['graph_y']]

    losses = []

    def _RecordLoss(epoch, data):
      """Callback to record training/prediction loss."""
      del epoch
      losses.append(data['loss'])

    callbacks = [keras.callbacks.LambdaCallback(on_epoch_end=_RecordLoss)]

    if log.type == 'train':
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
