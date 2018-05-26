"""The CLgen language model."""
import os
import pathlib
import typing

from absl import logging
from keras import callbacks
from keras import layers
from keras import models
from prettytable import PrettyTable

from deeplearning.clgen import cache
from deeplearning.clgen import corpus
from deeplearning.clgen import errors
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.proto import model_pb2
from lib.labm8 import crypto
from lib.labm8 import fs
from lib.labm8 import lockfile
from lib.labm8 import pbutil


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

    Raises:
      UserError: In case on an invalid config.
    """
    self.config = model_pb2.Model()
    self.config.CopyFrom(config)
    self.corpus = corpus.Corpus(config.corpus)
    self.hash = self._ComputeHash(self.corpus, self.config)
    self.cache = cache.mkcache('model', f'{self.corpus.language}-{self.hash}')
    self._model: typing.Optional[models.Sequential] = None
    logging.debug('model %s', self.hash)

    if not config.architecture.HasField('neuron_type'):
      raise errors.UserError('Model.archictecture.neuron_type specified.')

    # Validate metadata against cache.
    if self.cache.get('META.pbtxt'):
      cached_meta = pbutil.FromFile(pathlib.Path(self.cache['META.pbtxt']),
                                    internal_pb2.ModelMeta())
      # Exclude num_epochs from metadata comparison.
      config_to_compare = model_pb2.Model()
      config_to_compare.CopyFrom(self.config)
      config_to_compare.training.ClearField('num_epochs')
      cached_to_compare = model_pb2.Model()
      cached_to_compare.CopyFrom(cached_meta.config)
      cached_to_compare.training.ClearField('num_epochs')
      if config_to_compare != cached_to_compare:
        raise errors.InternalError('Metadata mismatch')
      self.meta = cached_meta
    else:
      self.meta = internal_pb2.ModelMeta()
      self.meta.config.CopyFrom(self.config)
      self._FlushMeta()

  @staticmethod
  def _ComputeHash(corpus_: corpus.Corpus, config: model_pb2.Model) -> str:
    """Compute model hash.

    The hash is computed from the ID of the corpus and the serialized
    representation of the config proto. The number of epochs that the model is
    trained for does not affect the hash, since we can share checkpoints
    between different models if the only variable is the epoch count. E.g.
    we have a model trained for 10 epochs, we can use the checkpoint as the
    starting point for a training a model for 20 epochs.
    """
    config_to_hash = model_pb2.Model()
    config_to_hash.CopyFrom(config)
    config_to_hash.ClearField('corpus')
    config_to_hash.training.ClearField('num_epochs')
    return crypto.sha1_list(corpus_.hash,
                            config_to_hash.SerializeToString())

  def InitAndGetTensorflow(self, infer: bool = False):
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
    seq_length = 1 if infer else self.corpus.sequence_length
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

  def _GetParamsPath(self, ckpt, num_batches: int) -> typing.Tuple[
    str, typing.List[str]]:
    """Return the path to checkpoint closest to target num of epochs."""
    paths = ckpt.all_model_checkpoint_paths
    batch_nums = [int(x.split('-')[-1]) for x in paths]
    epoch_nums = [int((x + 1) / num_batches) for x in batch_nums]

    closest = self.epochs
    closest_path = None
    for e, path in zip(epoch_nums, paths):
      diff = self.epochs - e
      if 0 <= diff < closest:
        logging.debug('  cached checkpoint at epoch = %d diff = %d', e, diff)
        closest = diff
        closest_path = path
    return closest_path, paths

  def GetKerasModel(self) -> models.Sequential:
    """Get the Keras model.

    If there is a cached model description, the model will be initialized from
    that. Else, it is constructed from the proto config. If there are cached
    weight checkpoints, the one closest to the target number of epochs will be
    loaded.

    Returns:
      A Sequential model instance.
    """
    if self.cache.get('model.yaml'):
      with open(self.cache['model.yaml']) as f:
        model = models.model_from_yaml(f.read())
      # TODO(cec): Load weights from checkpoint.
      return model
    else:
      model = BuildKerasModelFromProto(self.config,
                                       self.corpus.sequence_length,
                                       self.corpus.vocabulary_size)
      with open(self.cache.keypath('model.yaml'), 'w') as f:
        f.write(model.to_yaml())
      return model

  @property
  def model(self) -> models.Sequential:
    if self._model == None:
      self._model = self.GetKerasModel()
    else:
      return self._model

  def _LockedTrain(self) -> 'Model':
    """Locked training.

    Returns:
      The self instance.
    """
    # TODO(cec): Re-implement learning rate, decay rate, and gradient clip.
    learning_rate = self.config.training.initial_learning_rate
    decay_rate = self.config.training.percent_learning_rate_decay_per_epoch
    num_epochs = self.config.training.num_epochs
    batch_size = self.config.training.batch_size
    sequence_len = self.corpus.sequence_length

    X, y = self.corpus.GetTrainingData(shuffle=True)
    model = self.GetKerasModel()
    with open(self.cache.keypath('model.yaml'), 'w') as f:
      f.write(model.to_yaml())
    checkpoint_dir = pathlib.Path(self.cache.keypath('checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    file_path = str(
        checkpoint_dir) + "/checkpoint_weights_{epoch:02d}_{loss:.4f}.hdf5"
    checkpoint = callbacks.ModelCheckpoint(file_path, monitor="loss", verbose=1,
                                           save_best_only=True, mode="min")
    model.fit(X, y, epochs=num_epochs, batch_size=batch_size,
              callbacks=[checkpoint])
    # TODO(cec): Checkpoint callback.
    # stat = self.meta.training_stats.add()
    # stat.batch_num = batch_num + 1
    # stat.time_ms = int(epoch_duration * 1000)
    # stat.training_cost = float(train_cost)
    # self._FlushMeta()
    return self

  def _FlushMeta(self) -> None:
    pbutil.ToFile(self.meta, pathlib.Path(self.cache.keypath('META.pbtxt')))

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
  def most_recent_checkpoint_path(self) -> typing.Optional[str]:
    """Get path to most recent checkpoint, if exists.

    Returns:
      Path to the most recent checkpoint, or None if no checkpoints.
    """
    if self.cache.get("checkpoints"):
      last = sorted(pathlib.Path(self.cache['checkpoints']).iterdir())[-1]
      return fs.path(self.cache['checkpoints'], last)
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


def BuildKerasModelFromProto(config: model_pb2.Model,
                             sequence_length: int,
                             vocabulary_size: int) -> models.Sequential:
  """Build the Keras model from the proto config.

  Args:
    config: A Model proto instance.
    sequence_length: The length of sequences.
    vocabulary_size: The number of tokens in the vocabulary.

  Returns:
    A Sequential model instance.
  """
  model = models.Sequential()
  layer = {
    model_pb2.NetworkArchitecture.LSTM: layers.LSTM,
    model_pb2.NetworkArchitecture.RNN: layers.RNN,
    model_pb2.NetworkArchitecture.GRU: layers.GRU,
  }[config.architecture.neuron_type]
  return_sequences = config.architecture.neurons_per_layer > 1
  model.add(layer(config.architecture.neurons_per_layer,
                  input_shape=(
                    sequence_length,
                    vocabulary_size),
                  return_sequences=return_sequences))
  for _ in range(1, config.architecture.num_layers - 1):
    model.add(layer(config.architecture.neurons_per_layer,
                    return_sequences=True))
  model.add(layer(config.architecture.neurons_per_layer))
  model.add(layers.Dense(vocabulary_size))
  model.add(layers.Activation('softmax'))
  model.compile(loss='categorical_crossentropy', optimizer='adam')
  return model


def GetAllModels() -> typing.Iterator[Model]:
  """Iterate over all cached models.

  Returns:
    An iterable over all cached models.
  """
  if fs.isdir(cache.cachepath(), "model"):
    modeldirs = fs.ls(fs.path(cache.cachepath(), "model"), abspaths=True)
    for modeldir in modeldirs:
      meta = pbutil.FromFile(pathlib.Path(fs.path(modeldir, 'META.pbtxt')),
                             internal_pb2.ModelMeta())
      model = Model(meta.config)
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
