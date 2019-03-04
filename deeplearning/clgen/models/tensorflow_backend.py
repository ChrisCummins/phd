# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""CLgen models using a Keras backend."""
import os
import pathlib
import time
import typing

import humanize
import numpy as np
import progressbar
from absl import flags
from absl import logging

from deeplearning.clgen import samplers
from deeplearning.clgen import telemetry
from deeplearning.clgen.models import backends
from deeplearning.clgen.models import data_generators
from deeplearning.clgen.proto import model_pb2

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'clgen_tf_backend_reset_inference_state_between_batches', False,
    'If set, reset the network state between sample batches. Else, the model '
    'state is unaffected.')


class TensorFlowBackend(backends.BackendBase):
  """A model with an embedding layer, using a keras backend."""

  def __init__(self, *args, **kwargs):
    """Instantiate a model.

    Args:
      args: Arguments to be passed to BackendBase.__init__().
      kwargs: Arguments to be passed to BackendBase.__init__().
    """
    super(TensorFlowBackend, self).__init__(*args, **kwargs)

    # Attributes that will be lazily set.
    self.cell = None
    self.input_data = None
    self.targets = None
    self.lengths = None
    self.seed_length = None
    self.temperature = None
    self.initial_state = None
    self.logits = None
    self.generated = None
    self.loss = None
    self.final_state = None
    self.learning_rate = None
    self.epoch = None
    self.train_op = None

    self.inference_tf = None
    self.inference_sess = None
    self.first_sample = None
    self.inference_indices = None

  def InitTfGraph(self,
                  inference: bool,
                  inference_batch_size: typing.Optional[int] = None) -> 'tf':
    """Instantiate a TensorFlow graph for training or inference.

    The tensorflow graph is different for training and inference, so must be
    reset when switching between modes.

    Args:
      inference: If True, initialize model for inference. If False, initialize
        model for training.
      inference_batch_size: The batch size for inference.

    Returns:
      The imported TensorFlow module.
    """
    start_time = time.time()

    # Quiet tensorflow.
    # See: https://github.com/tensorflow/tensorflow/issues/1258
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Deferred importing of TensorFlow.
    import tensorflow as tf
    import tensorflow.contrib.seq2seq as seq2seq
    from tensorflow.contrib import rnn
    from deeplearning.clgen.models import helper

    cell_type = {
        model_pb2.NetworkArchitecture.LSTM: rnn.LSTMBlockCell,
        model_pb2.NetworkArchitecture.GRU: rnn.GRUBlockCellV2,
        model_pb2.NetworkArchitecture.RNN: rnn.BasicRNNCell,
    }.get(self.config.architecture.neuron_type, None)
    if cell_type is None:
      raise NotImplementedError

    # Reset the graph when switching between training and inference.
    tf.reset_default_graph()

    # Corpus attributes.
    sequence_length = 1024 if inference else self.config.training.sequence_length

    if inference:
      batch_size = inference_batch_size
    else:
      batch_size = self.config.training.batch_size
    vocab_size = self.atomizer.vocab_size

    cells_lst = []
    for _ in range(self.config.architecture.num_layers):
      cells_lst.append(cell_type(self.config.architecture.neurons_per_layer))
    self.cell = cell = rnn.MultiRNNCell(cells_lst, state_is_tuple=True)

    self.input_data = tf.placeholder(tf.int32, [batch_size, sequence_length])
    self.targets = tf.placeholder(tf.int32, [batch_size, sequence_length])
    self.initial_state = self.cell.zero_state(batch_size, tf.float32)
    self.temperature = tf.Variable(1.0, trainable=False)
    self.seed_length = tf.Variable(32, trainable=False)

    if inference:
      self.lengths = tf.placeholder(tf.int32, [batch_size])
    else:
      self.lengths = tf.fill([batch_size], sequence_length)

    scope_name = 'rnnlm'
    with tf.variable_scope(scope_name):
      with tf.device('/cpu:0'):
        embedding = tf.get_variable(
            'embedding',
            [vocab_size, self.config.architecture.neurons_per_layer])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

    if inference:
      decode_helper = helper.CustomInferenceHelper(
          inputs, self.lengths, self.seed_length, embedding, self.temperature)
    else:
      decode_helper = seq2seq.TrainingHelper(
          inputs, self.lengths, time_major=False)

    decoder = seq2seq.BasicDecoder(cell, decode_helper, self.initial_state,
                                   tf.layers.Dense(vocab_size))
    outputs, self.final_state, _ = seq2seq.dynamic_decode(
        decoder,
        output_time_major=False,
        impute_finished=True,
        swap_memory=True,
        scope=scope_name)

    self.generated = outputs.sample_id
    self.logits = outputs.rnn_output

    sequence_weigths = tf.ones(
        [self.config.training.batch_size, sequence_length])
    self.loss = seq2seq.sequence_loss(self.logits, self.targets,
                                      sequence_weigths)

    self.learning_rate = tf.Variable(0.0, trainable=False)
    self.epoch = tf.Variable(0, trainable=False)
    trainable_variables = tf.trainable_variables()

    # TODO(cec): Support non-adam optimizers.
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(self.loss, trainable_variables, aggregation_method=2),
        self.config.training.adam_optimizer.normalized_gradient_clip_micros /
        1e6)
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

    num_trainable_params = int(
        np.sum([np.prod(v.shape) for v in tf.trainable_variables()]))
    logging.info(
        'Instantiated TensorFlow graph with %s trainable parameters '
        'in %s ms.', humanize.intcomma(num_trainable_params),
        humanize.intcomma(int((time.time() - start_time) * 1000)))

    return tf

  @property
  def epoch_checkpoints(self) -> typing.Set[int]:
    """Get the set of epoch numbers which we have trained models for.

    Note that Tensorflow checkpoint paths don't translate to actual files, but
    rather a pair of <.index,.meta> files.

    Returns:
      A mapping of epoch numbers to paths.
    """
    if not (self.cache.path / 'checkpoints' / 'checkpoints'):
      # No saver file means no checkpoints.
      return {}

    # Count the number of checkpoint files which TensorFlow has created.
    checkpoint_files = [
        f.stem
        for f in (self.cache.path / 'checkpoints').iterdir()
        if f.name.startswith('checkpoint-') and f.name.endswith('.meta')
    ]
    # The checkpoint paths are appended with the epoch number.
    epoch_nums = [int(x.split('-')[-1]) for x in checkpoint_files]
    return set(epoch_nums)

  def GetParamsPath(self, checkpoint_state
                   ) -> typing.Tuple[typing.Optional[str], typing.List[str]]:
    """Return path to checkpoint closest to target num of epochs."""
    # Checkpoints are saved with relative path, so we must prepend cache paths.
    paths = [
        str(self.cache.path / 'checkpoints' / p)
        for p in checkpoint_state.all_model_checkpoint_paths
    ]
    # The checkpoint paths are appended with the epoch number.
    epoch_nums = [int(x.split('-')[-1]) for x in paths]
    diffs = [self.config.training.num_epochs - e for e in epoch_nums]
    pairs = zip(paths, diffs)
    positive_only = [p for p in pairs if p[1] >= 0]
    return min(positive_only, key=lambda x: x[1])[0], paths

  def InferenceManifest(self) -> typing.List[pathlib.Path]:
    """Return the list of files which are required for model inference.

    Returns:
      A list of absolute paths.
    """
    # The TensorFlow save file.
    paths = [
        self.cache.path / 'checkpoints' / 'checkpoint',
    ]
    # Export only the TensorFlow checkpoint files for the target number of
    # epochs.
    paths += [
        path.absolute()
        for path in (self.cache.path / 'checkpoints').iterdir()
        if path.name.startswith(f'checkpoint-{self.config.training.num_epochs}')
    ]
    # Include the epoch telemetry. This is not strictly required, but the files
    # are small and contain useful information for describing the model, such as
    # the total training time and model loss.
    paths += [
        path.absolute()
        for path in (self.cache.path / 'logs').iterdir()
        if (path.name.startswith('epoch_') and
            path.name.endswith('_telemetry.pbtxt'))
    ]
    return sorted(paths)

  def Train(self, corpus) -> None:
    """Locked training.

    If there are cached epoch checkpoints, the one closest to the target number
    of epochs will be loaded, and the model will be trained for only the
    remaining number of epochs, if any. This means that calling this function
    twice will only actually train the model the first time, and all subsequent
    calls will be no-ops.

    This method must only be called when the model is locked.
    """
    if self.is_trained:
      return

    data_generator = data_generators.TensorflowBatchGenerator(
        corpus, self.config.training)
    tf = self.InitTfGraph(inference=False)

    logger = telemetry.TrainingLogger(self.cache.path / 'logs')

    # training options
    # TODO(cec): Enable support for multiple optimizers:
    initial_learning_rate = (
        self.config.training.adam_optimizer.initial_learning_rate_micros / 1e6)
    decay_rate = (
        self.config.training.adam_optimizer.learning_rate_decay_per_epoch_micros
        / 1e6)

    # # resume from prior checkpoint
    ckpt_path, ckpt_paths = None, None
    if (self.cache.path / 'checkpoints' / 'checkpoint').exists():
      checkpoint_state = tf.train.get_checkpoint_state(
          self.cache.path / 'checkpoints')
      assert checkpoint_state
      assert checkpoint_state.model_checkpoint_path
      ckpt_path, ckpt_paths = self.GetParamsPath(checkpoint_state)

    with tf.Session() as sess:
      tf.global_variables_initializer().run()

      # Keep all checkpoints.
      saver = tf.train.Saver(
          tf.global_variables(), max_to_keep=100, save_relative_paths=True)

      # restore model from closest checkpoint.
      if ckpt_path:
        logging.info("Restoring checkpoint {}".format(ckpt_path))
        saver.restore(sess, ckpt_path)

      # make sure we don't lose track of other checkpoints
      if ckpt_paths:
        saver.recover_last_checkpoints(ckpt_paths)

      # Per-epoch training loop.
      for epoch_num in range(
          sess.run(self.epoch) + 1, self.config.training.num_epochs + 1):
        logger.EpochBeginCallback()

        # decay and set learning rate
        new_learning_rate = initial_learning_rate * (
            (float(100 - decay_rate) / 100.0)**(epoch_num - 1))
        sess.run(tf.assign(self.learning_rate, new_learning_rate))
        sess.run(tf.assign(self.epoch, epoch_num))

        # TODO(cec): refactor data generator to a Python generator.
        data_generator.CreateBatches()

        logging.info('Epoch %d/%d:', epoch_num, self.config.training.num_epochs)
        state = sess.run(self.initial_state)
        # Per-batch inner loop.
        bar = progressbar.ProgressBar(max_value=data_generator.num_batches)
        for _ in bar(range(data_generator.num_batches)):
          x, y = data_generator.NextBatch()
          feed = {self.input_data: x, self.targets: y}
          for i, (c, h) in enumerate(self.initial_state):
            feed[c] = state[i].c
            feed[h] = state[i].h
          loss, state, _ = sess.run(
              [self.loss, self.final_state, self.train_op], feed)

        # Log the loss and delta.
        logging.info('Loss: %.6f.', loss)

        # Save after every epoch.
        start_time = time.time()
        global_step = epoch_num
        checkpoint_prefix = (self.cache.path / 'checkpoints' / 'checkpoint')
        saver.save(sess, checkpoint_prefix, global_step=global_step)
        checkpoint_path = f'{checkpoint_prefix}-{global_step}'
        logging.info('Saved checkpoint %s in %s ms.', checkpoint_path,
                     humanize.intcomma(int((time.time() - start_time) * 1000)))
        assert pathlib.Path(
            f'{checkpoint_prefix}-{global_step}.index').is_file()
        assert pathlib.Path(f'{checkpoint_prefix}-{global_step}.meta').is_file()

        logger.EpochEndCallback(epoch_num, loss)

  def InitSampling(self,
                   sampler: samplers.Sampler,
                   seed: typing.Optional[int] = None) -> int:
    """Initialize model for sampling."""
    import tensorflow as tf
    # Delete any previous sampling session.
    if self.inference_tf:
      del self.inference_tf
    if self.inference_sess:
      del self.inference_sess

    self.inference_tf = self.InitTfGraph(
        inference=True, inference_batch_size=sampler.config.batch_size)
    self.inference_sess = self.inference_tf.Session()

    # Seed the RNG.
    if seed is not None:
      np.random.seed(seed)
      self.inference_tf.set_random_seed(seed)

    self.inference_tf.global_variables_initializer().run(
        session=self.inference_sess)
    # Restore trained model weights.
    saver = self.inference_tf.train.Saver(self.inference_tf.global_variables())
    checkpoint_state = self.inference_tf.train.get_checkpoint_state(
        self.cache.path / 'checkpoints')

    # These assertions will fail if the model has no checkpoints. Since this
    # should only ever be called after Train(), there is no good reason for
    # these assertions to fail.
    assert checkpoint_state
    assert checkpoint_state.model_checkpoint_path

    saver.restore(self.inference_sess, checkpoint_state.model_checkpoint_path)
    self.inference_sess.run(tf.assign(self.temperature, sampler.temperature))

    return sampler.config.batch_size

  def InitSampleBatch(self, sampler: samplers.Sampler, batch_size: int) -> None:
    if FLAGS.clgen_tf_backend_reset_inference_state_between_batches:
      self.inference_state = self.inference_sess.run(
          self.cell.zero_state(batch_size, self.inference_tf.float32))
    self.first_sample = True
    self.inference_indices = np.tile(sampler.encoded_start_text,
                                     [batch_size, 1])

  def SampleNextIndices(self, sampler: samplers.Sampler, batch_size: int,
                        done: np.ndarray):
    length = self.inference_indices.shape[1]
    assert length < 1024
    expanded_indices = np.zeros((batch_size, 1024))
    expanded_indices[:, :length] = self.inference_indices
    synthesized_lengths = np.full([batch_size], 1024)
    synthesized_lengths[done] = 0
    feed = {
        self.input_data: expanded_indices,
        self.lengths: synthesized_lengths,
        self.seed_length: length
    }

    generated, = self.inference_sess.run([self.generated], feed)
    self.inference_indices = generated[:, -1]
    if self.first_sample:
      generated = generated[:, len(sampler.encoded_start_text):]
    self.inference_indices = self.inference_indices.reshape((batch_size, 1))
    self.first_sample = False
    return generated

  @property
  def is_trained(self) -> bool:
    """Determine if model has been trained."""
    # Count the number of checkpoint files which TensorFlow has created.
    checkpoint_files = [
        f.stem
        for f in (self.cache.path / 'checkpoints').iterdir()
        if f.name.startswith('checkpoint-') and f.name.endswith('.meta')
    ]
    epoch_nums = [int(x.split('-')[-1]) for x in checkpoint_files]
    return self.config.training.num_epochs in epoch_nums
