"""Train and evaluate a model for graph-level classification."""
import typing
import warnings

import keras
import numpy as np
import tensorflow as tf
from keras import models

from deeplearning.ml4pl.graphs.labelled.graph_tuple import graph_batcher
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.lstm import bytecode2seq
from deeplearning.ml4pl.models.lstm import encoded_bytecode_database
from deeplearning.ml4pl.models.lstm import graph2seq
from deeplearning.ml4pl.models.lstm import lstm_utils as utils
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
app.DEFINE_integer("hidden_size", 64,
                   "The size of hidden layer(s) in the LSTM baselines.")
classifier_base.MODEL_FLAGS.add("hidden_size")

app.DEFINE_integer("dense_hidden_size", 32, "The size of the dense ")
classifier_base.MODEL_FLAGS.add("dense_hidden_size")

app.DEFINE_string('bytecode_encoder', 'inst2vec',
                  'The encoder to use. One of {llvm,inst2vec}')
classifier_base.MODEL_FLAGS.add("bytecode_encoder")

app.DEFINE_float('lang_model_loss_weight', .2,
                 'Weight for language model auxiliary loss.')
classifier_base.MODEL_FLAGS.add("lang_model_loss_weight")

app.DEFINE_database('encoded_bytecode_db', encoded_bytecode_database.Database,
                    None, 'Path to the encoded bytecode database to populate.')

app.DEFINE_string('group_by', 'statement', 'Group by statement or identifier.')

app.DEFINE_integer(
    "max_nodes_in_graph", 25000,
    "The maximum number of grouped statements to feed through "
    "the LSTM.")
#
##### End of flag declarations.


class LstmNodeClassifierModel(classifier_base.ClassifierBase):
  """LSTM baseline model for node level classification."""

  def __init__(self, *args, **kwargs):
    super(LstmNodeClassifierModel, self).__init__(*args, **kwargs)

    utils.SetAllowedGrowthOnKerasSession()

    # The encoder which performs translation from graphs to encoded sequences.
    if FLAGS.bytecode_encoder == 'llvm':
      bytecode_encoder = bytecode2seq.BytecodeEncoder()
    elif FLAGS.bytecode_encoder == 'inst2vec':
      bytecode_encoder = bytecode2seq.Inst2VecEncoder()
    else:
      raise app.UsageError(f"Unknown encoder '{FLAGS.bytecode_encoder}'")

    # The encoder which performs translation from graphs to encoded sequences.
    self.encoder = graph2seq.GraphToBytecodeGroupingsEncoder(
        self.batcher.db, bytecode_encoder, group_by=FLAGS.group_by)

    self.encoded_bytecode_db: encoded_bytecode_database.EncodedBytecode = (
        FLAGS.encoded_bytecode_db())

    # Language model

    # define token ids as input

    input_layer = keras.Input(
        batch_shape=(FLAGS.batch_size,
                     self.encoder.bytecode_encoder.max_sequence_length),
        dtype='int32',
        name="model_in")
    # and the segment indices
    input_segments = keras.Input(
        batch_shape=(FLAGS.batch_size,
                     self.encoder.bytecode_encoder.max_sequence_length),
        dtype='int32',
        name="model_in_segments")

    # lookup token embeddings
    encoded_inputs = keras.layers.Embedding(
        input_dim=self.encoder.bytecode_encoder.
        vocabulary_size_with_padding_token,
        input_length=self.encoder.bytecode_encoder.max_sequence_length,
        output_dim=FLAGS.hidden_size,
        name="embedding")(input_layer)

    # do the unsorted segment sum to get the actual lstm inputs
    def segment_sum_wrapper(args):
      """Sum the encoded_tokens by their segment IDs.

      Args:
        encoded_tokens.  Shape: (batch_size, sequence_length, embedding_dim).
        segment_ids.  Shape: (batch_size, segment_ids).

      Returns:
        Summed embedding vectors of Shape (batch_size, max_segment_id)
      """
      encoded_tokens, segment_ids = args

      segment_ids = tf.cast(segment_ids, dtype=tf.int32)
      max_segment_id = tf.cast(tf.math.reduce_max(segment_ids) + 1,
                               dtype=tf.int32)

      # Perform a segment sum for each row in the batch independently.
      segment_sums = [
          # Note the slice so that graphs larger than max_nodes_in_graph are
          # truncated.
          tf.math.unsorted_segment_sum(data=encoded_tokens[i],
                                       segment_ids=segment_ids[i],
                                       num_segments=max_segment_id)
          [:FLAGS.max_nodes_in_graph] for i in range(FLAGS.batch_size)
      ]

      return tf.stack(segment_sums, axis=0)

    # Encoded sequences input.
    x = keras.layers.Lambda(segment_sum_wrapper)(
        [encoded_inputs, input_segments])

    # Concatenate the "selector vector" input.
    selector_vector = keras.Input(batch_shape=(FLAGS.batch_size, None, 2),
                                  name="selector_vector")
    x = keras.layers.Concatenate()([x, selector_vector],)

    x = utils.MakeLstm(FLAGS.hidden_size, return_sequences=True,
                       name="lstm_1")(x)

    x = utils.MakeLstm(FLAGS.hidden_size,
                       return_sequences=True,
                       return_state=False,
                       name="lstm_2")(x)

    # map to number of classes with a dense layer
    out = keras.layers.Dense(2, activation="sigmoid", name="out")(x)

    # pass both inputs to the model class.
    self.model = keras.Model(
        inputs=[input_layer, input_segments, selector_vector], outputs=[out])
    # self.model.summary()
    # this is a hack to make the predictions during training accessible:
    # https://github.com/keras-team/keras/issues/3469
    # train_on_batch will now return a second argument.
    # self.model.metrics_tensors = [out]
    self.model.compile(optimizer="adam",
                       metrics=['accuracy'],
                       loss=["categorical_crossentropy"],
                       loss_weights=[1.0])

  def GetEncodedBytecodes(
      self, bytecode_ids: typing.List[int]
  ) -> typing.Tuple[typing.List[np.array], typing.List[np.array], typing.
                    List[np.array]]:
    """Get the encoded bytecodes.

    Args:
      bytecode_ids: A list of bytecode IDs to encode.

    Returns:
      A tuple of encoded_sequences, segment_ids, and node_masks.
    """
    unique_bytecode_ids = set(bytecode_ids)

    # Get the set of bytecode IDs that have already been processed.
    with self.encoded_bytecode_db.Session() as session:
      query = session.query(encoded_bytecode_database.EncodedBytecode)
      query = query.filter(
          encoded_bytecode_database.EncodedBytecode.bytecode_id.in_(
              unique_bytecode_ids))
      id_to_encoded_sequences: typing.Dict[int, np.array] = ({
          row.bytecode_id: row.encoded_sequence for row in query
      })
      id_to_segment_ids: typing.Dict[int, np.array] = ({
          row.bytecode_id: row.segment_ids for row in query
      })
      id_to_node_mask: typing.Dict[int, np.array] = ({
          row.bytecode_id: row.node_mask for row in query
      })

    # Encode any bytecodes not already found in the database.
    bytecode_ids_to_encode = list(unique_bytecode_ids -
                                  id_to_encoded_sequences.keys())
    if bytecode_ids_to_encode:
      with prof.Profile(f'Encoded {len(bytecode_ids_to_encode)} bytecodes',
                        print_to=lambda x: app.Log(2, x)):
        encoded_sequences, grouping_ids, node_masks = self.encoder.EncodeBytecodes(
            bytecode_ids_to_encode)

      with prof.Profile(f'Storing {len(encoded_sequences)} encoded sequences',
                        print_to=lambda x: app.Log(2, x)):
        with self.encoded_bytecode_db.Session(commit=True) as session:
          for bytecode_id, encoded_sequence, segment_ids, node_mask in zip(
              bytecode_ids_to_encode, encoded_sequences, grouping_ids,
              node_masks):
            result = encoded_bytecode_database.EncodedBytecode(
                bytecode_id=bytecode_id)
            result.encoded_sequence = encoded_sequence
            result.segment_ids = segment_ids
            result.node_mask = node_mask
            id_to_encoded_sequences[bytecode_id] = encoded_sequence
            id_to_segment_ids[bytecode_id] = segment_ids
            id_to_node_mask[bytecode_id] = node_mask
            session.add(result)

    return (
        [id_to_encoded_sequences[bytecode_id] for bytecode_id in bytecode_ids],
        [id_to_segment_ids[bytecode_id] for bytecode_id in bytecode_ids],
        [id_to_node_mask[bytecode_id] for bytecode_id in bytecode_ids],
    )

  def MakeMinibatchIterator(
      self, epoch_type: str, groups: typing.List[str], print_context: typing.Any = None
  ) -> typing.Iterable[typing.Tuple[log_database.BatchLogMeta, typing.Any]]:
    """Create minibatches by encoding, padding, and concatenating text
    sequences."""
    if FLAGS.batch_size > 1024:
      raise ValueError(
          f"Here batch size counts number of graphs, so {FLAGS.batch_size} is "
          f"too many.")
    options = graph_batcher.GraphBatchOptions(max_graphs=FLAGS.batch_size,
                                              groups=groups)
    max_instance_count = (
        FLAGS.max_train_per_epoch if epoch_type == 'train' else
        FLAGS.max_val_per_epoch if epoch_type == 'val' else None)
    for batch in self.batcher.MakeGraphBatchIterator(options,
                                                     max_instance_count,
                                                     print_context):
      # Get the encoded bytecodes.
      bytecode_ids = batch.log._transient_data['bytecode_ids']
      encoded_sequences, segment_ids, node_masks = self.GetEncodedBytecodes(
          bytecode_ids)

      # takes node_y and node_x_indices and splits it
      # such that the resulting lists hold one graph of the batch per element
      # so batch_size is the len of the resulting lists
      split_indices = np.where(
          batch.graph_nodes_list[:-1] != batch.graph_nodes_list[1:])[0] + 1
      all_node_y_per_graph = np.split(batch.node_y, split_indices)
      all_node_x_per_graph = np.split(batch.node_x_indices, split_indices)

      # confirm that we have as many node_masks as batch_size
      if len(node_masks) != len(all_node_y_per_graph):
        raise OSError(f"len(node_masks)={len(node_masks)} != "
                      f"len(all_node_y_per_graph)={len(all_node_y_per_graph)}")
      if len(node_masks) != len(all_node_x_per_graph):
        raise OSError(f"len(node_masks)={len(node_masks)} != "
                      f"len(all_node_x_per_graph)={len(all_node_x_per_graph)}")

      # Mask only the "active" node labels.
      # Shape (batch_size, ?, 2)
      node_y_per_graph = [
          node_y[np.where(node_mask)]
          for node_y, node_mask in zip(all_node_y_per_graph, node_masks)
      ]

      max_nodes_in_graph = max(max(m) for m in segment_ids) + 1
      # Enforce a maximum length on the number of statements that can be
      # processed to prevent OOM for really big graphs.
      max_nodes_in_graph = min(max_nodes_in_graph, FLAGS.max_nodes_in_graph)
      app.Log(2, "Padding graph batch to %s nodes", max_nodes_in_graph)

      # Shape (batch_size, max_nodes_in_graph, 2)
      node_y_truncated = np.array(
          keras.preprocessing.sequence.pad_sequences(
              node_y_per_graph,
              maxlen=max_nodes_in_graph,
              value=np.array([0, 0], dtype=np.int32),
          ))

      # Select the second node x embedding indices as a one hot vector.
      # Shape (batch_size, ?, 2)
      selector_vectors = [
          node_x[node_mask]
          for node_x, node_mask in zip(all_node_x_per_graph, node_masks)
      ]

      # Shape (batch_size, max_nodes_in_graph, 2)
      selector_vectors = np.array(
          keras.preprocessing.sequence.pad_sequences(
              selector_vectors,
              maxlen=max_nodes_in_graph,
              value=np.array([0, 0], dtype=np.int32),
          ))

      assert selector_vectors.shape == node_y_truncated.shape

      # Pad and truncate encoded sequences
      # Shape (batch_size, max_sequence_length)
      encoded_sequences = np.array(
          keras.preprocessing.sequence.pad_sequences(
              encoded_sequences,
              maxlen=self.encoder.bytecode_encoder.max_sequence_length,
              value=self.encoder.bytecode_encoder.pad_val,
          ))
      # Shape (batch_size, max_sequence_length)
      segment_ids = np.array(
          keras.preprocessing.sequence.pad_sequences(
              segment_ids,
              maxlen=self.encoder.bytecode_encoder.max_sequence_length,
              value=max_nodes_in_graph,
          ))

      yield batch.log, {
          # Shape (batch_size, max_sequence_length)
          'encoded_sequences': np.vstack(encoded_sequences),
          # Shape (batch_size, max_sequence_length)
          'segment_ids': np.vstack(segment_ids),
          # Shape (batch_size, max_nodes_in_graph)
          'selector_vectors': selector_vectors,
          # The y values as fed to the model.
          # Shape (batch_size, max_nodes_in_graph, 2)
          'node_y_truncated': node_y_truncated,
          # Shape (batch_size, ?, 2).
          'node_y': node_y_per_graph,
      }

  def RunMinibatch(self, log: log_database.BatchLogMeta, batch: typing.Any
                  ) -> classifier_base.ClassifierBase.MinibatchResults:
    """Run a batch through the LSTM."""
    log.loss = 0

    x = [
        batch['encoded_sequences'],
        batch['segment_ids'],
        batch['selector_vectors'],
    ]
    y = [batch['node_y_truncated']]

    with prof.Profile(f'model.train_on_batch() {len(y[0])} instances',
                      print_to=lambda x: app.Log(2, x)):
      if log.type == 'train':
        loss = self.model.train_on_batch(x, y)[0]
        log.loss = loss

      pred_y = self.model.predict_on_batch(x)

    assert len(pred_y) == len(
        batch['node_y']
    ), f"len pred_y={len(pred_y)} and len(batch)={len(batch['node_y'])}"

    num_classes = pred_y.shape[-1]

    # To handle the fact that LSTMs can't always receive the entire input
    # sequence, we pad the predictions.
    padded_y_pred = []
    for y_pred_for_graph, true_node_y in zip(pred_y, batch['node_y']):
      pad_count = max(len(true_node_y) - len(y_pred_for_graph), 0)
      if pad_count > 0:
        padding = np.zeros((pad_count, num_classes))
        padded_y_pred.append(np.concatenate([y_pred_for_graph, padding]))
      else:
        padded_y_pred.append(y_pred_for_graph[:len(true_node_y)])

    # Flatten the per-graph predictions and labels.
    pred_y = np.concatenate(padded_y_pred)
    y_true = np.concatenate(batch['node_y'])

    return self.MinibatchResults(y_true_1hot=y_true, y_pred_1hot=pred_y)

  def ModelDataToSave(self):
    model_path = self.working_dir / f'{self.run_id}_keras_model.h5'
    self.model.save(model_path)
    return {'model_path': model_path}

  def LoadModelData(self, data_to_load: typing.Any):
    model_path = data_to_load['model_path']
    models.load_model(model_path)


def main():
  """Main entry point."""
  # TODO(cec): Only filter https://scikit-learn.org/stable/modules/generated/sklearn.exceptions.UndefinedMetricWarning.html
  warnings.filterwarnings("ignore")
  classifier_base.Run(LstmNodeClassifierModel)


if __name__ == '__main__':
  app.Run(main)
