"""Train and evaluate a model for graph-level classification."""
import typing

import keras
import numpy as np
import tensorflow as tf
from keras import models

from deeplearning.ml4pl.graphs.labelled.graph_tuple import graph_batcher
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.lstm import graph2seq
from labm8 import app

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
app.DEFINE_integer("hidden_size", 200,
                   "The size of hidden layer(s) in the LSTM baselines.")
classifier_base.MODEL_FLAGS.add("hidden_size")

app.DEFINE_integer("dense_hidden_size", 32, "The size of the dense ")
classifier_base.MODEL_FLAGS.add("dense_hidden_size")

app.DEFINE_float('lang_model_loss_weight', .2,
                 'Weight for language model auxiliary loss.')
classifier_base.MODEL_FLAGS.add("lang_model_loss_weight")
#
##### End of flag declarations.


class LstmNodeClassifierModel(classifier_base.ClassifierBase):
  """LSTM baseline model for node level classification."""

  def __init__(self, *args, **kwargs):
    super(LstmNodeClassifierModel, self).__init__(*args, **kwargs)

    # The encoder which performs translation from graphs to encoded sequences.
    self.encoder = graph2seq.GraphToSequenceEncoder(self.batcher.db)

    # Language model

    # define token ids as input

    input_layer = keras.Input(
        batch_shape=(FLAGS.batch_size, self.encoder.max_sequence_length),
        dtype='int32',
        name="model_in")
    # and the segment indices
    input_segments = keras.Input(
        batch_shape=(FLAGS.batch_size, self.encoder.max_sequence_length),
        dtype='int32',
        name="model_in_segments")

    # input_graph_node_list = keras.Input(
    #     batch_shape=(FLAGS.batch_size, self.encoder.max_sequence_length),
    #     dtype='int32',
    #     name='graph_node_list_input')

    # lookup token embeddings
    encoded_inputs = keras.layers.Embedding(
        input_dim=self.encoder.vocabulary_size_with_padding_token,
        input_length=self.encoder.max_sequence_length,
        output_dim=FLAGS.hidden_size,
        name="embedding")(input_layer)

    # do the unsorted segment sum to get the actual lstm inputs
    def segment_sum_wrapper(args):
      """Sum the encoded_tokens by their segment IDs.

      Args:
        encoded_tokens.  Shape: (batch_size, sequence_length, embedding_dim).
        segment_ids.  Shape: (batch_size, segment_ids).
      """
      encoded_tokens, segment_ids = args

      segment_ids = tf.cast(segment_ids, dtype=tf.int32)
      max_segment_id = tf.cast(tf.math.reduce_max(segment_ids) + 1,
                               dtype=tf.int32)

      # Perform a segment sum for each row in the batch independently.
      segment_sums = [
          tf.math.unsorted_segment_sum(data=encoded_tokens[i],
                                       segment_ids=segment_ids[i],
                                       num_segments=max_segment_id)
          for i in range(FLAGS.batch_size)
      ]

      return tf.stack(segment_sums, axis=0)

    x = keras.layers.Lambda(segment_sum_wrapper)(
        [encoded_inputs, input_segments])

    # vanilla
    x = keras.layers.CuDNNLSTM(FLAGS.hidden_size,
                               return_sequences=True,
                               name="lstm_1")(x)

    x = keras.layers.CuDNNLSTM(FLAGS.hidden_size,
                               name="lstm_2",
                               return_sequences=True,
                               return_state=False)(x)

    # map to number of classes with a dense layer
    langmodel_out = keras.layers.Dense(self.stats.node_labels_dimensionality,
                                       activation="sigmoid",
                                       name="langmodel_out")(x)

    # no graph level features for node classification.
    out = langmodel_out

    # pass both inputs to the model class.
    self.model = keras.Model(inputs=[input_layer, input_segments],
                             outputs=[out])
    #self.model.summary()

    self.model.compile(optimizer="adam",
                       metrics=['accuracy'],
                       loss=["categorical_crossentropy"],
                       loss_weights=[1.0])

  def MakeMinibatchIterator(
      self, epoch_type: str, group: str
  ) -> typing.Iterable[typing.Tuple[log_database.BatchLogMeta, typing.Any]]:
    """Create minibatches by encoding, padding, and concatenating text
    sequences."""
    if FLAGS.batch_size > 1024:
      raise ValueError(
          f"Here batch size counts number of graphs, so {FLAGS.batch_size} is too many."
      )

    options = graph_batcher.GraphBatchOptions(max_graphs=FLAGS.batch_size,
                                              group=group)
    max_instance_count = (
        FLAGS.max_train_per_epoch if epoch_type == 'train' else
        FLAGS.max_val_per_epoch if epoch_type == 'val' else None)
    for batch in self.batcher.MakeGraphBatchIterator(options,
                                                     max_instance_count):
      graph_ids = batch.log._graph_indices
      encoded_sequences, grouping_ids, node_masks = (
          self.encoder.GraphsToEncodedStatementGroups(graph_ids,
                                                      group_by='statement'))

      assert batch.node_y is not None
      yield batch.log, {
          'encoded_sequences': np.vstack(encoded_sequences),
          'segment_ids': np.vstack(grouping_ids),
          'node_x_indices': np.vstack(batch.node_x_indices),
          # TODO(cec): what to do with node_masks?
          'node_y': np.vstack(batch.node_y),
      }

  def RunMinibatch(self, log: log_database.BatchLogMeta, batch: typing.Any
                  ) -> classifier_base.ClassifierBase.MinibatchResults:
    """Run a batch through the LSTM."""
    log.loss = 0

    x = [
        batch['encoded_sequences'],
        batch['segment_ids'],
    ]
    # TODO(cec): Mask node_y
    y = [batch['node_y']]

    losses = []

    def _RecordLoss(epoch, data):
      """Callback to record training/prediction loss."""
      del epoch
      app.Log(2, 'Loss %s', data['loss'])
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

    # TODO
    # Run the same input again through the LSTM to get the raw predictions.
    # This is obviously wasteful when training, but I don't know of a way to
    # get the raw predictions from self.model.fit().
    pred_y = self.model.predict(x)[0]

    return batch.graph_y, pred_y

  def ModelDataToSave(self):
    model_path = self.working_dir / f'{self.run_id}_keras_model.h5'
    self.model.save(model_path)
    return {'model_path': model_path}

  def LoadModelData(self, data_to_load: typing.Any):
    model_path = data_to_load['model_path']
    models.load_model(model_path)


def main():
  """Main entry point."""
  classifier_base.Run(LstmNodeClassifierModel)


if __name__ == '__main__':
  app.Run(main)
