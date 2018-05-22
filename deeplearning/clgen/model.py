"""The CLgen language model."""
import os
import typing
from time import time

import progressbar
from absl import logging
from prettytable import PrettyTable

from deeplearning.clgen import cache
from deeplearning.clgen import corpus
from deeplearning.clgen import errors
from deeplearning.clgen.proto import model_pb2
from lib.labm8 import crypto
from lib.labm8 import fs
from lib.labm8 import jsonutil
from lib.labm8 import lockfile


class Model(object):
  """
  A CLgen Model.

  Please note model instances should be treated as immutable. Upon
  instantiation, a model's properties are used to determine its hash. If you
  modify a property after instantiation, the hash will be out of date, which
  can lead to bad things happening.
  """

  def __init__(self, config: model_pb2.Model):
    """Instantiate a model.

    Args:
      config: A Model message.
    """
    self.config = model_pb2.Model()
    self.config.CopyFrom(config)
    self.corpus = corpus.Corpus(config.corpus)
    self.hash = self._ComputeHash(self.corpus, self.config)
    self.cache = cache.mkcache('model', f'{self.corpus.language}-{self.hash}')
    logging.debug('model %s', self.hash)

    if not config.architecture.HasField('neuron_type'):
      raise errors.UserError('Model.archictecture.neuron_type specified.')

    # validate metadata against cache, and restore stats
    self.stats = {'epoch_times': [], 'epoch_costs': [], 'epoch_batches': []}
    self.meta = {"stats": self.stats}
    if self.cache.get("META"):
      cached_meta = jsonutil.read_file(self.cache["META"])
      self.stats = cached_meta["stats"]  # restore stats

      if "stats" in cached_meta:
        del cached_meta["stats"]
      del self.meta["stats"]

      if self.meta != cached_meta:
        logging.error("Computed META: %s", jsonutil.format_json(self.meta))
        raise errors.InternalError(
          "metadata mismatch in model %s" % self.cache["META"])
    else:
      self._FlushMeta()

  @staticmethod
  def _ComputeHash(corpus_: corpus.Corpus, config: model_pb2.Model) -> str:
    """Compute model hash.

    The hash is computed from the ID of the corpus and the serialized
    representation of the config proto.
    """
    config_without_corpus = model_pb2.Model()
    config_without_corpus.CopyFrom(config)
    config_without_corpus.ClearField('corpus')
    return crypto.sha1_list(corpus_.hash,
                            config_without_corpus.SerializeToString())

  def _InitAndGetTensorflow(self, infer: bool = False):
    """Import Tensorflow runtime and return it.

    Deferred importing of tensorflow and initializing model for training
    or sampling.

    This is necessary for two reasons: first, the tensorflow graph is
    different for training and inference, so must be reset when switching
    between modes. Second, importing tensorflow takes a long time, so
    we only want to do it if we actually need to.

    Args:
      infer: If True, initialize model for inference. If False, initialize
        model for training.

    Returns:
      TensorFlow module.
    """
    # Quiet TensorFlow. See:
    # https://github.com/tensorflow/tensorflow/issues/1258
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    import tensorflow as tf
    from tensorflow.contrib import legacy_seq2seq as seq2seq
    from tensorflow.contrib import rnn

    self.cell_fn = {model_pb2.NetworkArchitecture.LSTM: rnn.BasicLSTMCell,
                    model_pb2.NetworkArchitecture.GRU: rnn.GRUCell,
                    model_pb2.NetworkArchitecture.RNN: rnn.BasicRNNCell}.get(
      self.config.architecture.neuron_type)
    # Reset the graph when switching between training and inference.
    tf.reset_default_graph()
    batch_size = self.config.training.batch_size
    # Corpus info:
    seq_length = 1 if infer else self.corpus.seq_length
    vocab_size = self.corpus.vocab_size
    # Model state:
    cell = self.cell_fn(self.neurons_per_layer, state_is_tuple=True)
    self.cell = cell = rnn.MultiRNNCell([cell] * self.num_layers,
                                        state_is_tuple=True)
    self.input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
    self.targets = tf.placeholder(tf.int32, [batch_size, seq_length])
    self.initial_state = self.cell.zero_state(batch_size, tf.float32)
    scope_name = 'rnnlm'
    with tf.variable_scope(scope_name):
      softmax_w = tf.get_variable("softmax_w",
                                  [self.neurons_per_layer, vocab_size])
      softmax_b = tf.get_variable("softmax_b", [vocab_size])

      with tf.device("/cpu:0"):
        embedding = tf.get_variable("embedding",
                                    [vocab_size, self.neurons_per_layer])
        inputs = tf.split(axis=1, num_or_size_splits=seq_length,
                          value=tf.nn.embedding_lookup(embedding,
                                                       self.input_data))
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

    def InferenceLoop(prev, _):
      prev = tf.matmul(prev, softmax_w) + softmax_b
      prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
      return tf.nn.embedding_lookup(embedding, prev_symbol)

    outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell,
                                              loop_function=InferenceLoop if
                                              infer else None,
                                              scope=scope_name)
    output = tf.reshape(tf.concat(axis=1, values=outputs),
                        [-1, self.neurons_per_layer])
    self.logits = tf.matmul(output, softmax_w) + softmax_b
    self.probs = tf.nn.softmax(self.logits)
    loss = seq2seq.sequence_loss_by_example([self.logits],
                                            [tf.reshape(self.targets, [-1])], [
                                              tf.ones(
                                                [batch_size * seq_length])],
                                            vocab_size)
    self.cost = tf.reduce_sum(loss) / batch_size / seq_length
    self.final_state = last_state
    self.learning_rate = tf.Variable(0.0, trainable=False)
    self.epoch = tf.Variable(0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(  # Argument of potential interest:
      #   aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE
      #
      # See:
      #   https://www.tensorflow.org/api_docs/python/tf/gradients
      #   https://www.tensorflow.org/api_docs/python/tf/AggregationMethod
      tf.gradients(self.cost, tvars), self.gradient_clip)
    optimizer = tf.train.AdamOptimizer(self.learning_rate)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    return tf

  def _GetParamsPath(self, ckpt) -> typing.Tuple[str, typing.List[str]]:
    """Return the path to checkpoint closest to target num of epochs."""
    paths = ckpt.all_model_checkpoint_paths
    batch_nums = [int(x.split('-')[-1]) for x in paths]
    epoch_nums = [int((x + 1) / (self.corpus.num_batches)) for x in batch_nums]

    closest = self.epochs
    closest_path = None
    for e, path in zip(epoch_nums, paths):
      diff = self.epochs - e
      if 0 <= diff < closest:
        logging.debug('  cached checkpoint at epoch = %d diff = %d', e, diff)
        closest = diff
        closest_path = path
    return closest_path, paths

  def _LockedTrain(self) -> 'Model':
    tf = self._InitAndGetTensorflow(infer=False)

    # training options
    learning_rate = self.config.training.initial_learning_rate
    decay_rate = self.config.training.percent_learning_rate_decay_per_epoch

    # resume from prior checkpoint
    checkpoint_path, checkpoint_paths = None, None
    if self.most_recent_checkpoint_path:
      # Check that all necessary files exist.
      assert fs.isdir(self.most_recent_checkpoint_path)
      checkpoint = tf.train.get_checkpoint_state(
        self.most_recent_checkpoint_path)
      assert checkpoint
      assert checkpoint.model_checkpoint_path
      checkpoint_path, checkpoint_paths = self._GetParamsPath(checkpoint)

    with tf.Session() as sess:
      tf.global_variables_initializer().run()

      # Keep all checkpoints.
      saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

      # Restore model from closest checkpoint.
      if checkpoint_path:
        logging.debug('restoring %s', checkpoint_path)
        saver.restore(sess, checkpoint_path)
        logging.debug('restored checkpoint %s', checkpoint_path)

      # make sure we don't lose track of other checkpoints
      if checkpoint_paths:
        saver.recover_last_checkpoints(checkpoint_paths)

      max_batch = self.epochs * self.corpus.num_batches

      bar = progressbar.ProgressBar(max_value=max_batch)
      if sess.run(self.epoch) != self.epochs:
        logging.info('training %s', self)

      for e in range(sess.run(self.epoch) + 1, self.epochs + 1):
        epoch_start = time()

        # Decay and set learning rate.
        new_learning_rate = learning_rate * (
            (float(100 - decay_rate) / 100.0) ** (e - 1))
        sess.run(tf.assign(self.learning_rate, new_learning_rate))
        sess.run(tf.assign(self.epoch, e))
        self.corpus.CreateBatches(self.config.training.batch_size,
                                  self.config.training.shuffle_corpus_contentfiles_between_epochs)
        state = sess.run(self.initial_state)
        for b in range(self.corpus.num_batches):
          x, y = self.corpus.next_batch()
          feed = {self.input_data: x, self.targets: y}
          for i, (c, h) in enumerate(self.initial_state):
            feed[c] = state[i].c
            feed[h] = state[i].h
          train_cost, state, _ = sess.run(
            [self.cost, self.final_state, self.train_op], feed)

          # update progress bar
          batch_num = (e - 1) * self.corpus.num_batches + b
          bar.update(batch_num)

        # Determine whether we should save a checkpoint.
        should_save = self.config.training.save_intermediate_checkpoints
        # Always save on last epoch.
        should_save |= e == self.epochs
        if should_save:
          saver.save(sess, self.cache.keypath("model.ckpt"),
                     global_step=batch_num)

          next_checkpoint = e * self.corpus.num_batches + b
          max_epoch = self.epochs
          logging.debug('\n%s epoch %d / %d. next checkpoint at batch %d', self,
                        e, max_epoch, next_checkpoint)

          # Update training time.
          epoch_duration = time() - epoch_start
          self.stats["epoch_costs"].append(float(train_cost))
          self.stats["epoch_times"].append(epoch_duration)
          self.stats["epoch_batches"].append(batch_num + 1)
          self._FlushMeta()
    return self

  def _FlushMeta(self) -> None:
    jsonutil.write_file(self.cache.keypath("META"), self.meta)

  def Train(self) -> 'Model':
    """Train the model.

    Returns:
      The model instance.
    """
    with self.lock.acquire(replace_stale=True):
      return self._LockedTrain()

  @property
  def shorthash(self):
    return cache.ShortHash(self.hash, cache.cachepath("model"))

  @property
  def lock(self) -> lockfile.LockFile:
    lockpath = self.cache.keypath("LOCK")
    return lockfile.LockFile(lockpath)

  @property
  def neuron_type(self) -> model_pb2.NetworkArchitecture.NeuronType:
    return self.config.architecture.neuron_type

  @property
  def neurons_per_layer(self) -> int:
    return self.config.architecture.neurons_per_layer

  @property
  def num_layers(self) -> int:
    return self.config.architecture.num_layers

  @property
  def gradient_clip(self) -> int:
    return self.config.training.gradient_clip

  @property
  def epochs(self) -> int:
    return self.config.training.num_epochs

  @property
  def most_recent_checkpoint_path(self) -> typing.Optional[str]:
    """Get path to most recent checkpoint, if exists.

    Returns:
      Path to the most recent checkpoint, or None if no checkpoints.
    """
    if self.cache.get("checkpoint"):
      return self.cache.path
    else:
      return None

  def __repr__(self) -> str:
    """String representation."""
    celltype = model_pb2.NetworkArchitecture.NeuronType.Name(self.neuron_type)
    return (f'model[{self.shorthash}]: '
            f'{self.neurons_per_layer}x{self.num_layers}x{self.epochs} '
            f'{celltype}')

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, Model):
      return False
    return rhs.hash == self.hash

  def __ne__(self, rhs) -> bool:
    return not self.__eq__(rhs)


def GetAllModels() -> typing.Iterator[Model]:
  """Iterate over all cached models.

  Returns:
    An iterable over all cached models.
  """
  if fs.isdir(cache.cachepath(), "model"):
    modeldirs = fs.ls(fs.path(cache.cachepath(), "model"), abspaths=True)
    for modeldir in modeldirs:
      meta = jsonutil.read_file(fs.path(modeldir, "META"))
      # TODO(cec): Fix.
      model = Model(meta)
      yield model


def ModelsToTable(*models: typing.List[Model]) -> PrettyTable:
  """
  Pretty print a table of model stats.

  Parameters
  ----------
  models : typing.List[Model]
      Models to tablify.

  Returns
  -------
  PrettyTable
      Formatted table for printing.
  """
  tab = PrettyTable(
    ["model", "corpus", "trained", "type", "nodes", "epochs", "lr", "dr",
     "gc", ])

  tab.align['nodes'] = 'r'
  tab.sortby = "nodes"

  for model in models:
    meta = model.meta

    nodes = meta["architecture"]["rnn_size"]
    layers = meta["architecture"]["num_layers"]

    if "stats" in meta:
      num_epochs = len(meta["stats"]["epoch_costs"])
    else:
      num_epochs = 0

    if num_epochs >= meta["train_opts"]["epochs"]:
      trained = "Y"
    elif fs.isfile(fs.path(model.cache.path, "LOCK")):
      trained = f"WIP ({num_epochs}/{meta['train_opts']['epochs']})"
    elif num_epochs > 0:
      trained = f"{num_epochs}/{meta['train_opts']['epochs']}"
    else:
      trained = ""

    tab.add_row([model.shorthash, model.corpus.shorthash, trained,
                 meta["architecture"]["model_type"], f'{nodes} x {layers}',
                 meta["train_opts"]["epochs"],
                 "{:.0e}".format(meta["train_opts"]["learning_rate"]),
                 meta["train_opts"]["lr_decay_rate"],
                 meta["train_opts"]["grad_clip"], ])

  return tab
