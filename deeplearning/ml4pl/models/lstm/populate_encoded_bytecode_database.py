"""TODO."""
import os
import typing

import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from deeplearning.ml4pl.graphs.labelled.graph_tuple import graph_batcher
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.lstm import bytecode2seq
from deeplearning.ml4pl.models.lstm import encoded_bytecode_database
from deeplearning.ml4pl.models.lstm import graph2seq
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

app.DEFINE_string('bytecode_encoder', 'llvm',
                  'The encoder to use. One of {llvm,inst2vec}')
classifier_base.MODEL_FLAGS.add("bytecode_encoder")

app.DEFINE_float('lang_model_loss_weight', .2,
                 'Weight for language model auxiliary loss.')
classifier_base.MODEL_FLAGS.add("lang_model_loss_weight")

app.DEFINE_database('encoded_bytecode_db', encoded_bytecode_database.Database,
                    None, 'Path to the encoded bytecode database to populate.')
#
##### End of flag declarations.


class EncodedBytecodePopulator(classifier_base.ClassifierBase):
  """TODO."""

  def __init__(self, *args, **kwargs):
    super(EncodedBytecodePopulator, self).__init__(*args, **kwargs)

    self.encoded_bytecode_db: encoded_bytecode_database.Database = (
        FLAGS.encoded_bytecode_db())

    # The encoder which performs translation from graphs to encoded sequences.
    if FLAGS.bytecode_encoder == 'llvm':
      bytecode_encoder = bytecode2seq.BytecodeEncoder()
    elif FLAGS.bytecode_encoder == 'inst2vec':
      bytecode_encoder = bytecode2seq.Inst2VecEncoder()
    else:
      raise app.UsageError(f"Unknown encoder '{FLAGS.bytecode_encoder}'")

    # The encoder which performs translation from graphs to encoded sequences.
    self.encoder = graph2seq.GraphToBytecodeGroupingsEncoder(
        self.batcher.db, bytecode_encoder, group_by='statement')

  def MakeMinibatchIterator(
      self, epoch_type: str, groups: typing.List[str]
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
                                                     max_instance_count):
      bytecode_ids = batch.log._transient_data['bytecode_ids']
      bytecode_ids = list(set(bytecode_ids))

      with prof.Profile(f'self.encoder.Encode() {len(bytecode_ids)} bytecodes:',
                        print_to=lambda x: app.Log(2, x)):
        encoded_sequences, grouping_ids, node_masks = self.encoder.EncodeBytecodes(
            bytecode_ids)

      with prof.Profile(f'Storing {len(encoded_sequences)} encoded sequences'):
        with self.encoded_bytecode_db.Session(commit=True) as session:
          # TODO: Query this earlier so that we don't do unnecessary encoding.
          query = session.query(
              encoded_bytecode_database.EncodedBytecode.bytecode_id)
          query = query.filter(
              encoded_bytecode_database.EncodedBytecode.bytecode_id.in_(
                  bytecode_ids))
          already_done = set([row.bytecode_id for row in query])
          if already_done:
            app.Warning('Ignoring %s already-done bytecodes', len(already_done))

          for bytecode_id, encoded_sequence, segment_ids, node_mask in zip(
              bytecode_ids, encoded_sequences, grouping_ids, node_masks):
            if bytecode_id in already_done:
              continue
            result = encoded_bytecode_database.EncodedBytecode(
                bytecode_id=bytecode_id)
            result.encoded_sequence = encoded_sequence
            result.segment_ids = segment_ids
            result.node_mask = node_masks
            session.add(result)

      yield batch.log, {}

  def RunMinibatch(self, log: log_database.BatchLogMeta, batch: typing.Any
                  ) -> classifier_base.ClassifierBase.MinibatchResults:
    """Run a batch through the LSTM."""
    log.loss = 0
    return self.MinibatchResults(y_true_1hot=np.array([[0, 1]]),
                                 y_pred_1hot=np.array([[1, 0]]))

  def ModelDataToSave(self):
    return {}

  def LoadModelData(self, data_to_load: typing.Any):
    pass


def main():
  """Main entry point."""
  classifier_base.Run(EncodedBytecodePopulator)


if __name__ == '__main__':
  app.Run(main)
