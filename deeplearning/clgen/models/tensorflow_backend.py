"""CLgen models using a Keras backend."""
import os
import typing

import humanize
import numpy as np
import progressbar
from absl import flags
from absl import logging

from deeplearning.clgen import samplers
from deeplearning.clgen import telemetry
from deeplearning.clgen.models import data_generators
from deeplearning.clgen.models import models
from deeplearning.clgen.proto import model_pb2
from lib.labm8 import crypto
from lib.labm8 import labdate
from lib.labm8 import logutil
from lib.labm8 import pbutil


FLAGS = flags.FLAGS


class TensorFlowModel(models.ModelBase):
  """A model with an embedding layer, using a keras backend."""

  def __init__(self, config: model_pb2.Model):
    """Instantiate a model.

    Args:
      config: A Model message.

    Raises:
      TypeError: If the config argument is not a Model proto.
      UserError: In case on an invalid config.
    """
    super(TensorFlowModel, self).__init__(config)

    # Attributes that will be lazily set.
    self._training_model: typing.Optional['keras.models.Sequential'] = None
    self._inference_model: typing.Optional['keras.models.Sequential'] = None
    self._inference_batch_size: typing.Optional[int] = None

  def InitTfGraph(self, inference: bool) -> 'tf':
    """Instantiate a TensorFlow graph for training or inference.

    The tensorflow graph is different for training and inference, so must be
    reset when switching between modes.

    Args:
      inference: If True, initialize model for inference. If False, initialize
        model for training.

    Returns:
      The imported TensorFlow module.
    """
    self.corpus.Create()

    # Quiet tensorflow.
    # See: https://github.com/tensorflow/tensorflow/issues/1258
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Deferred importing of TensorFlow.
    import tensorflow as tf
    import tensorflow.contrib.legacy_seq2seq as seq2seq
    from tensorflow.contrib import rnn

    cell_type = {
      model_pb2.NetworkArchitecture.LSTM: rnn.BasicLSTMCell,
      model_pb2.NetworkArchitecture.GRU: rnn.GRUCell,
      model_pb2.NetworkArchitecture.RNN: rnn.BasicRNNCell,
    }.get(self.config.architecture.neuron_type, None)
    if cell_type is None:
      raise NotImplementedError

    # Reset the graph when switching between training and inference.
    tf.reset_default_graph()

    # Corpus attributes.
    sequence_length = 1 if inference else self.config.training.sequence_length
    self.data_generator = data_generators.TensorflowBatchGenerator(
        self.corpus, self.config.training)

    vocab_size = self.corpus.vocab_size

    cell = cell_type(
        self.config.architecture.neurons_per_layer, state_is_tuple=True)
    self.cell = cell = rnn.MultiRNNCell(
        [cell] * self.config.architecture.num_layers, state_is_tuple=True)
    self.input_data = tf.placeholder(
        tf.int32, [self.config.training.batch_size, sequence_length])
    self.targets = tf.placeholder(
        tf.int32, [self.config.training.batch_size, sequence_length])
    self.initial_state = self.cell.zero_state(
        self.config.training.batch_size, tf.float32)

    scope_name = 'rnnlm'
    with tf.variable_scope(scope_name):
      softmax_w = tf.get_variable(
          'softmax_w', [self.config.architecture.neurons_per_layer, vocab_size])
      softmax_b = tf.get_variable('softmax_b', [vocab_size])

      with tf.device('/cpu:0'):
        embedding = tf.get_variable(
            'embedding',
            [vocab_size, self.config.architecture.neurons_per_layer])
        inputs = tf.split(
            axis=1, num_or_size_splits=sequence_length,
            value=tf.nn.embedding_lookup(embedding, self.input_data))
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    def InferenceLoop(prev, _):
      prev = tf.matmul(prev, softmax_w) + softmax_b
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      return tf.nn.embedding_lookup(embedding, prev_symbol)

    outputs, last_state = seq2seq.rnn_decoder(
        inputs, self.initial_state, cell, scope=scope_name,
        loop_function=InferenceLoop if inference else None)
    output = tf.reshape(tf.concat(axis=1, values=outputs),
                        [-1, self.config.architecture.neurons_per_layer])
    self.logits = tf.matmul(output, softmax_w) + softmax_b
    self.probs = tf.nn.softmax(self.logits)
    sequence_loss = seq2seq.sequence_loss_by_example(
        [self.logits],
        [tf.reshape(self.targets, [-1])],
        [tf.ones([self.config.training.batch_size * sequence_length])],
        vocab_size)
    self.loss = tf.reduce_sum(
        sequence_loss) / self.config.training.batch_size / sequence_length
    self.final_state = last_state
    self.learning_rate = tf.Variable(0.0, trainable=False)
    self.epoch = tf.Variable(0, trainable=False)
    trainable_variables = tf.trainable_variables()
    # TODO(cec): Support non-adam optimizers.
    grads, _ = tf.clip_by_global_norm(
        # Argument of potential interest:
        #   aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE
        #
        # See:
        #   https://www.tensorflow.org/api_docs/python/tf/gradients
        #   https://www.tensorflow.org/api_docs/python/tf/AggregationMethod
        tf.gradients(self.loss, trainable_variables),
        self.config.training.adam_optimizer.normalized_gradient_clip_micros /
        1e6)
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.train_op = optimizer.apply_gradients(zip(grads, trainable_variables))

    return tf

  def GetTrainedModel(self) -> None:
    """Train TensorFlow model."""
    with self.lock.acquire(replace_stale=True, block=True):
      self._LockedTrain()
    total_time_ms = sum(
        t.epoch_wall_time_ms
        for t in self.TrainingTelemetry()[:self.config.training.num_epochs])
    logging.info('Trained model for %d epochs in %s ms (%s).',
                 self.config.training.num_epochs,
                 humanize.intcomma(total_time_ms),
                 humanize.naturaldelta(total_time_ms / 1000))
    return None

  def GetParamsPath(self, checkpoint_state) -> typing.Tuple[
    typing.Optional[str], typing.List[str]]:
    """Return path to checkpoint closest to target num of epochs."""
    paths = checkpoint_state.all_model_checkpoint_paths
    # The checkpoint paths are appended with the epoch number.
    epoch_nums = [int(x.split('-')[-1]) for x in paths]
    diffs = [self.config.training.num_epochs - e for e in epoch_nums]
    pairs = zip(paths, diffs)
    positive_only = [p for p in pairs if p[1] >= 0]
    return min(positive_only, key=lambda x: x[1])[0], paths

  def _LockedTrain(self) -> None:
    """Locked training.

    If there are cached epoch checkpoints, the one closest to the target number
    of epochs will be loaded, and the model will be trained for only the
    remaining number of epochs, if any. This means that calling this function
    twice will only actually train the model the first time, and all subsequent
    calls will be no-ops.

    This method must only be called when the model is locked.
    """
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
      saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

      # restore model from closest checkpoint.
      if ckpt_path:
        saver.restore(sess, ckpt_path)
        logging.info("restored checkpoint {}".format(ckpt_path))

      # make sure we don't lose track of other checkpoints
      if ckpt_paths:
        saver.recover_last_checkpoints(ckpt_paths)

      max_batch = self.config.training.num_epochs * self.data_generator.num_batches

      if sess.run(self.epoch) != self.config.training.num_epochs:
        logging.info("training, %s", self)

      # Per-epoch training loop.
      # TODO(cec): Per-epoch progress bars.
      for epoch_num in range(sess.run(self.epoch) + 1,
                             self.config.training.num_epochs + 1):
        logger.EpochBeginCallback()

        # decay and set learning rate
        new_learning_rate = initial_learning_rate * (
            (float(100 - decay_rate) / 100.0) ** (epoch_num - 1))
        sess.run(tf.assign(self.learning_rate, new_learning_rate))
        sess.run(tf.assign(self.epoch, epoch_num))

        # TODO(cec): refactor data generator to a generator.
        self.data_generator.CreateBatches()

        logging.info('Epoch %d/%d:', epoch_num, self.config.training.num_epochs)
        state = sess.run(self.initial_state)
        # Per-batch inner loop.
        bar = progressbar.ProgressBar(max_value=self.data_generator.num_batches)
        for _ in bar(range(self.data_generator.num_batches)):
          x, y = self.data_generator.NextBatch()
          feed = {self.input_data: x, self.targets: y}
          for i, (c, h) in enumerate(self.initial_state):
            feed[c] = state[i].c
            feed[h] = state[i].h
          loss, state, _ = sess.run(
              [self.loss, self.final_state, self.train_op], feed)

        logging.info('Loss: %.6f', loss)
        # Save after every epoch.
        global_step = epoch_num
        checkpoint_prefix = (self.cache.path / 'checkpoints' / 'checkpoint')
        saver.save(sess, checkpoint_prefix, global_step=global_step)
        checkpoint_path = f'{checkpoint_prefix}-{global_step}'
        logging.info(f'Saved file to {checkpoint_path}')
        # TODO(cec): Assert .meta and .index files have been created.

        logger.EpochEndCallback(epoch_num, loss)

  def Sample(self, sampler: samplers.Sampler,
             min_num_samples: int,
             seed: int = None) -> typing.List[model_pb2.Sample]:
    """Sample a model.

    If the model is not already trained, calling Sample() first trains the
    model. Thus a call to Sample() is equivalent to calling Train() then
    Sample().

    Args:
      sampler: The sampler to sample using.
      min_num_samples: The minimum number of samples to return. Note that the
        true number of samples returned may be higher than this value, as
        sampling occurs in batches. The model will continue producing samples
        until the lowest mulitple of the sampler batch size property that is
        larger than this value. E.g. if min_num_samples is 7 and the Sampler
        batch size is 10, 10 samples will be returned.

    Returns:
      A list of Sample protos.

    Raises:
      UnableToAcquireLockError: If the model is locked (i.e. there is another
        process currently modifying the model).
      InvalidStartText: If the sampler start text cannot be encoded.
      InvalidSymtokTokens: If the sampler symmetrical depth tokens cannot be
        encoded.
    """
    self.Train()

    sample_count = 1
    self.SamplerCache(sampler).mkdir(exist_ok=True)
    with logutil.TeeLogsToFile(
        f'sampler_{sampler.hash}', self.cache.path / 'logs'):
      logging.info("Sampling: '%s'", sampler.start_text)
      if min_num_samples < 0:
        logging.warning(
            'Entering an infinite sample loop, this process will never end!')
      sample_start_time = labdate.MillisecondsTimestamp()

      tf = self.InitTfGraph(inference=True)

      sampler.Specialize(self.corpus.atomizer)
      samples = []

      batch_size = self.config.training.batch_size

      # Seed the RNG.
      if seed is not None:
        np.random.seed(seed)
        tf.set_random_seed(seed)

      with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        checkpoint_state = tf.train.get_checkpoint_state(
            self.cache.path / 'checkpoints')

        # These assertions will fail if the model has no checkpoints. Since this
        # method first calls Train(), there is no good reason for these
        # assertions to fail.
        assert checkpoint_state
        assert checkpoint_state.model_checkpoint_path

        saver.restore(sess, checkpoint_state.model_checkpoint_path)

        def weighted_pick(weights):
          """
          requires that all probabilities are >= 0, i.e.:
            assert all(x >= 0 for x in weights)
          See: https://github.com/ChrisCummins/clgen/issues/120
          """
          t = np.cumsum(weights)
          s = np.sum(weights)
          return int(np.searchsorted(t, np.random.rand(1) * s))

        # Per-sample batch outer loop.
        while True:
          samples_in_progress = [
            sampler.tokenized_start_text.copy()
            for _ in range(batch_size)]
          done = np.zeros(batch_size, dtype=np.bool)
          start_time = labdate.MillisecondsTimestamp()
          wall_time_start = start_time

          state = sess.run(self.cell.zero_state(batch_size, tf.float32))
          indices = np.zeros((batch_size, 1))

          # Seed the model state with the starting text.
          for symbol in sampler.encoded_start_text[:-1]:
            indices[:] = symbol
            feed = {self.input_data: indices, self.initial_state: state}
            [state] = sess.run([self.final_state], feed)
          indices[:] = sampler.encoded_start_text[-1]

          # Sample-batch inner loop.
          while True:
            feed = {self.input_data: indices, self.initial_state: state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)

            # sample distribution to pick next symbol:
            indices[:, 0] = [weighted_pick(p) for p in probs]

            for i in range(batch_size):
              if done[i]:
                continue

              token = self.corpus.atomizer.decoder[indices[i, 0]]
              samples_in_progress[i].append(token)
              if sampler.SampleIsComplete(samples_in_progress[i]):
                end_time = labdate.MillisecondsTimestamp()
                done[i] = 1
                sample = model_pb2.Sample(
                    text=''.join(samples_in_progress[i]),
                    sample_start_epoch_ms_utc=start_time,
                    sample_time_ms=end_time - start_time,
                    wall_time_ms=end_time - wall_time_start,
                    num_tokens=len(samples_in_progress[i]))
                print(f'=== BEGIN CLGEN SAMPLE {sample_count} '
                      f'===\n\n{sample.text}\n')
                sample_count += 1
                sample_id = crypto.sha256_str(sample.text)
                sample_path = self.SamplerCache(sampler) / f'{sample_id}.pbtxt'
                pbutil.ToFile(sample, sample_path)
                if min_num_samples > 0:
                  samples.append(sample)
                wall_time_start = labdate.MillisecondsTimestamp()

            # Complete the batch.
            if done.all():
              break

          # Complete sampling.
          if len(samples) >= min_num_samples:
            now = labdate.MillisecondsTimestamp()
            logging.info(
                'Produced %s samples at a rate of %s ms / sample.',
                humanize.intcomma(len(samples)),
                humanize.intcomma(
                    int((now - sample_start_time) / len(samples))))
            break

    return samples
