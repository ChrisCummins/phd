"""The CLgen language model."""
import io
import pathlib
import queue
import sys
import threading
import typing

import humanize
import numpy as np
import progressbar
from absl import logging
from keras import callbacks
from keras import models
from keras import utils
from prettytable import PrettyTable

from deeplearning.clgen import cache
from deeplearning.clgen import corpuses
from deeplearning.clgen import errors
from deeplearning.clgen import languages
from deeplearning.clgen import samplers
from deeplearning.clgen.models import builders
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from lib.labm8 import cache as labcache
from lib.labm8 import crypto
from lib.labm8 import fs
from lib.labm8 import labdate
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
      TypeError: If the config argument is not a Model proto.
      UserError: In case on an invalid config.
    """
    # Error early, so that a cache isn't created.
    if not isinstance(config, model_pb2.Model):
      t = type(config).__name__
      raise TypeError(f"Config must be a Model proto. Received: '{t}'")

    if not config.architecture.HasField('neuron_type'):
      raise errors.UserError('Model.architecture.neuron_type field not set')

    # Attributes that will be lazily set.
    self._model: typing.Optional[models.Sequential] = None
    self._current_weights_epoch: int = 0

    self.config = model_pb2.Model()
    self.config.CopyFrom(config)
    self.corpus = corpuses.Corpus(config.corpus)
    self.hash = self._ComputeHash(self.corpus, self.config)
    self.cache = cache.mkcache('model', f'{self.corpus.language}-{self.hash}')
    # Create the necessary cache directories.
    (self.cache.path / 'checkpoints').mkdir(exist_ok=True)
    (self.cache.path / 'samples').mkdir(exist_ok=True)
    logging.debug('model %s', self.hash)

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
      self._WriteMetafile()

  @staticmethod
  def _ComputeHash(corpus_: corpuses.Corpus, config: model_pb2.Model) -> str:
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

  def GetKerasModel(self) -> models.Sequential:
    """Get the Keras model.

    If there is a cached model description, the model will be initialized from
    that. Else, it is constructed from the proto config.

    Returns:
      A Sequential model instance.
    """
    if self.cache.get('model.yaml'):
      with open(self.cache['model.yaml']) as f:
        model = models.model_from_yaml(f.read())
        model.compile(loss='categorical_crossentropy', optimizer='adam')
      return model
    else:
      model = builders.BuildKerasModel(self.config, self.corpus.sequence_length,
                                       self.corpus.vocabulary_size)
      with open(self.cache.keypath('model.yaml'), 'w') as f:
        f.write(model.to_yaml())
      return model

  @property
  def model(self) -> models.Sequential:
    if self._model is None:
      self._model = self.GetKerasModel()
    return self._model

  def _LockedTrain(self) -> 'Model':
    """Locked training.

    If there are cached epoch checkpoints, the one closest to the target number
    of epochs will be loaded, and the model will be trained for only the
    remaining number of epochs, if any. This means that calling this function
    twice will only actually train the model the first time, and all subsequent
    calls will be no-ops.

    This method must only be called when the model is locked.

    Returns:
      The Model instance (self).
    """
    target_num_epochs = self.config.training.num_epochs
    starting_epoch = 0

    # Early exit in case the model is already sufficiently trained.
    if target_num_epochs == self._current_weights_epoch:
      return

    epoch_checkpoints = self.epoch_checkpoints
    if len(epoch_checkpoints) >= target_num_epochs:
      # We have already trained a model to at least this number of epochs, so
      # simply the weights from that epoch and call it a day.
      self.model.load_weights(
          epoch_checkpoints[target_num_epochs - 1])
      return
    elif epoch_checkpoints:
      # We have already trained a model at least part of the way to our target
      # number of epochs, so load the most recent one.
      self.model.load_weights(epoch_checkpoints[-1])
      starting_epoch = len(epoch_checkpoints)

    # model.load_weights(self.most_recent_checkpoint_path)
    # TODO(cec): Re-implement learning rate, decay rate, and gradient clip.
    learning_rate = self.config.training.initial_learning_rate
    decay_rate = self.config.training.percent_learning_rate_decay_per_epoch

    checkpoint_dir = pathlib.Path(self.cache.keypath('checkpoints'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    file_path = str(
        checkpoint_dir) + "/checkpoint_weights_{epoch:02d}_{loss:.4f}.hdf5"
    checkpoint = callbacks.ModelCheckpoint(file_path, monitor="loss", verbose=1,
                                           save_best_only=False, mode="min")
    generator = DataGenerator(
        self.corpus, self.config.training.batch_size,
        self.corpus.sequence_length,
        self.config.training.shuffle_corpus_contentfiles_between_epochs)
    logging.info('Steps per epoch: %s',
                 humanize.intcomma(generator.steps_per_epoch))
    self.model.fit_generator(generator,
                             steps_per_epoch=generator.steps_per_epoch,
                             epochs=target_num_epochs - starting_epoch,
                             callbacks=[checkpoint])
    self._current_weights_epoch = self.config.training.num_epochs
    # TODO(cec): Checkpoint callback.
    # stat = self.meta.training_stats.add()
    # stat.batch_num = batch_num + 1
    # stat.time_ms = int(epoch_duration * 1000)
    # stat.training_cost = float(train_cost)
    # self._WriteMetafile()
    return self

  def _LockedSample(self, sampler: samplers.Sampler, min_num_samples: int) -> \
      typing.List[internal_pb2.Sample]:
    """Locked sampling.

    This method must only be called when the model is locked.

    Args:
      sampler: A Sampler instance.
      min_num_samples: The minimum number of samples to return. If -1, the
        sampler runs indefinitely, and this method never returns.

    Returns:
      A list of samples.

    Raises:
      InvalidStartText: If the sampler's start text cannot be encoded using the
        corpus vocabulary.
    """
    logging.info('Sampling %s', sampler)
    try:
      encoded_seed = self.corpus.atomizer.AtomizeString(sampler.start_text)
    except errors.VocabError:
      raise errors.InvalidStartText(
          'Sampler start text cannot be encoded using the corpus vocabulary: '
          f"'{sampler.start_text}'")

    samples = []

    # TODO(cec): Re-implement batched sampling.
    vectorized_seed = np.zeros(
        (1, self.corpus.sequence_length, self.corpus.vocabulary_size),
        dtype=np.bool)
    for i, token in enumerate(encoded_seed):
      vectorized_seed[0, i, token] = 1

    while len(samples) < min_num_samples:
      X = np.copy(vectorized_seed)
      text = []
      start_time = labdate.MillisecondsTimestamp()
      # Save a few cycles by not going via the model() property every time we need
      # to access it.
      model = self.model
      # TODO(cec): Re-implement sampling termination criteria.
      for i in range(100):
        prediction = np.argmax(model.predict(X, verbose=0))
        text.append(self.corpus.atomizer.decoder[prediction])
        activations = np.zeros((1, 1, self.corpus.vocabulary_size),
                               dtype=np.bool)
        activations[0, 0, prediction] = 1
        X = np.concatenate((X[:, 1:, :], activations), axis=1)
      end_time = labdate.MillisecondsTimestamp()
      sample = internal_pb2.Sample(text=''.join(text),
                                   sample_start_epoch_ms_utc=start_time,
                                   sample_time_ms=end_time - start_time)
      sample_id = crypto.sha256_str(sample.text)
      p = self.SamplerCache(sampler) / f'{sample_id}.pbtxt'
      pbutil.ToFile(sample, p)
      samples.append(sample)

    return samples

  def Train(self) -> 'Model':
    """Train the model.

    Returns:
      The model instance.
    """
    with self.lock.acquire(replace_stale=True):
      return self._LockedTrain()

  def Sample(self, sampler: samplers.Sampler,
             min_num_samples: int) -> typing.List[internal_pb2.Sample]:
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
    """
    self.SamplerCache(sampler).mkdir(exist_ok=True)
    with self.lock.acquire(replace_stale=True):
      self.Train()
      return self._LockedSample(sampler, min_num_samples)

  def SamplerCache(self, sampler: samplers.Sampler) -> pathlib.Path:
    """Get the path to a sampler cache.

    Args:
      sampler: A Sampler instance.

    Returns:
      A path to a directory. Note that this directory may not exist.
    """
    return self.cache.path / 'samples' / sampler.hash

  def _WriteMetafile(self) -> None:
    pbutil.ToFile(self.meta, pathlib.Path(self.cache.keypath('META.pbtxt')))

  @property
  def shorthash(self) -> str:
    return cache.ShortHash(self.hash, cache.cachepath("model"))

  @property
  def is_trained(self) -> bool:
    """Return whether the model has previously been trained."""
    return len(self.epoch_checkpoints) >= self.config.training.num_epochs

  @property
  def lock(self) -> lockfile.LockFile:
    lockpath = self.cache.keypath("LOCK")
    return lockfile.LockFile(lockpath)

  @property
  def epoch_checkpoints(self) -> typing.List[pathlib.Path]:
    """Get the paths to all epoch checkpoint files in order.

    Remember that the returned list is zero-indexed, so the epoch number is
    the array index plus one. E.g. The checkpoint for epoch 5 is
    epoch_checkpoints[4].

    Returns:
      A list of paths.
    """
    checkpoint_dir = pathlib.Path(self.cache.path) / 'checkpoints'
    return [checkpoint_dir / x for x in
            sorted(pathlib.Path(self.cache['checkpoints']).iterdir())]

  def __repr__(self) -> str:
    """String representation."""
    celltype = model_pb2.NetworkArchitecture.NeuronType.Name(
        self.config.architecture.neuron_type)
    return (f'model[{self.shorthash}]: '
            f'{self.neurons_per_layer}x{self.num_layers}x{self.epochs} '
            f'{celltype}')

  def __eq__(self, rhs) -> bool:
    if not isinstance(rhs, Model):
      return False
    return rhs.hash == self.hash

  def __ne__(self, rhs) -> bool:
    return not self.__eq__(rhs)


class DataGenerator(object):
  """Generated X, y training data pairs of one-hot encoded text.

  We train our network on overlapping one-hot encoded text sequences. For a
  corpus of a reasonable size, this won't fit in memory. This class provides
  a generator for use by a sequential Keras model's fit_generator() method to
  feed in training data.
  """

  def __init__(self, corpus_: corpuses.Corpus, batch_size: int,
               sequence_length: int, shuffle: bool):
    self.corpus = corpus_
    self.batch_size = batch_size
    self.encoded_corpus = self.corpus.GetTrainingData(shuffle=shuffle)
    self.corpus_len = len(self.encoded_corpus)
    self.sequence_length = sequence_length
    self.skip = 1  # TODO(cec): Add this as a field in Model proto.
    self.shuffle = shuffle

    # Set this publicly visibly attribute. The number of steps per epoch is
    # the total number of batches per epoch.
    self.steps_per_epoch = int(
        self.corpus_len / (self.batch_size * self.sequence_length))

    self.i = 0

  def __next__(self) -> typing.Tuple[np.ndarray, np.ndarray]:
    """Generate the next batch of X, y pairs."""
    if self.i + self.batch_size + self.sequence_length >= self.corpus_len:
      self.i = 0
      if self.shuffle:
        self.encoded_corpus = self.corpus.GetTrainingData(shuffle=True)

    X_data = []
    y_data = []
    for i in range(self.i, self.i + self.batch_size, self.skip):
      sequence = self.encoded_corpus[i:i + self.sequence_length]
      next_token = self.encoded_corpus[i + self.sequence_length]
      X_data.append(sequence)
      y_data.append(next_token)

    num_sentences = len(X_data)
    assert num_sentences == self.batch_size
    logging.debug('Sliced %d sequences of length %d', num_sentences,
                 self.sequence_length)
    # Vectorize.
    X = np.zeros(
        (num_sentences, self.sequence_length, self.corpus.vocabulary_size),
        dtype=np.bool)
    y = np.zeros((num_sentences, self.corpus.vocabulary_size), dtype=np.bool)
    for i, sequence in enumerate(X_data):
      for t, encoded_char in enumerate(sequence):
        X[i, t, encoded_char] = 1
      y[i, y_data[i]] = 1

    # TODO(cec): Use keras to_categorical() instead of vectorizing by hand.
    _ = utils.to_categorical(y_data, self.corpus.vocabulary_size)

    self.i += self.batch_size
    return X, y


class SampleProducer(threading.Thread):
  def __init__(self, model: Model, sampler_config: sampler_pb2.Sampler,
               q: queue.Queue):
    super(SampleProducer, self).__init__()
    self.model = model
    self.sampler_config = sampler_config
    self.queue = q
    self.stop_signal = threading.Event()
    self.sample_header = '\n\n' + languages.format_as_comment(
        self.model.corpus.language, '==== START SAMPLE ====') + '\n\n'

    # Determine the termination criteria.
    self.max_length = -1
    self.special_token_left = None
    self.special_token_right = None
    for criterion in self.sampler_config.termination_criteria:
      if criterion.HasField('maxlen'):
        self.max_length = criterion.maxlen.maximum_tokens_in_sample
        if not criterion.maxlen.include_start_text_in_maximum:
          self.max_length += len(self.sampler_config.start_text)
      elif criterion.HasField('symtok'):
        self.symmetrical_token_left = criterion.symtok.depth_increase_token
        self.symmetrical_token_right = criterion.symtok.depth_decrease_token
    self.has_max_length = self.max_length > 0
    self.has_symmetrical_tokens = (
        self.special_token_left and self.special_token_right)

  def run(self) -> None:
    batch_size = self.sampler_config.batch_size

    # Fail if the model is locked. Models are locked during training.
    if self.model.lock.islocked:
      raise lockfile.UnableToAcquireLockError(self.lock)

    tf = self.model.InitAndGetTensorflow(infer=True)

    # Seed the RNG.
    np.random.seed(self.sampler_config.seed)
    tf.set_random_seed(self.sampler_config.seed)

    with tf.Session() as sess:
      tf.global_variables_initializer().run()
      saver = tf.train.Saver(tf.global_variables())
      checkpoint = tf.train.get_checkpoint_state(self.model.cache.path)
      # Sanity checks.
      assert checkpoint
      assert checkpoint.model_checkpoint_path
      saver.restore(sess, checkpoint.model_checkpoint_path)

      def WeightedPick(weights):
        """Make a weighted choice.

        Requires that all probabilities are >= 0, i.e.:
          assert all(x >= 0 for x in weights)
        See: https://github.com/ChrisCummins/clgen/issues/120
        """
        t = np.cumsum(weights)
        s = np.sum(weights)
        return int(np.searchsorted(t, np.random.rand(1) * s))

      def GetSymmetricalTokenDepth(text: str, depth: int) -> typing.Tuple[
        int, int]:
        """Calculate the sample depth for symmetrical tokens."""
        depth += text.count(self.symmetrical_token_left)
        is_started = depth > 0
        depth -= text.count(self.symmetrical_token_right)
        return is_started, depth

      if self.has_symmetrical_tokens:
        init_started, init_depth = GetSymmetricalTokenDepth(
            self.sampler_config.start_text, 0)
      atomize = self.model.corpus.atomizer.AtomizeString
      deatomize = self.model.corpus.atomizer.DeatomizeIndices

      while not self.stop_requested:
        buf = [io.StringIO() for _ in range(batch_size)]
        if self.has_symmetrical_tokens:
          depth = [init_depth] * batch_size
          started = [init_started] * batch_size
        running = [True] * batch_size

        state = sess.run(self.model.cell.zero_state(batch_size, tf.float32))
        indices = np.zeros((batch_size, 1))

        seed_tensor = atomize(self.sampler_config.start_text)
        for symbol in seed_tensor[:-1]:
          indices[:] = symbol
          feed = {self.model.input_data: indices,
                  self.model.initial_state: state}
          [state] = sess.run([self.model.final_state], feed)
        for item in range(batch_size):
          buf[item].write(self.sampler_config.start_text)
        indices[:] = seed_tensor[-1]
        i = 0
        while True:
          feed = {self.model.input_data: indices,
                  self.model.initial_state: state}
          try:
            [probs, state] = sess.run(
                [self.model.probs, self.model.final_state], feed)
          except tf_errors.InvalidArgumentError:
            logging.warning('sampling error')
            self.run()
          # Sample distribution to pick next symbols:
          indices[:, 0] = [WeightedPick(p) for p in probs]

          for item in range(batch_size):
            if not running[item]:
              continue

            # In case of decoding error, start sampling again:
            # try:
            atom = deatomize([indices[item, 0]])
            # except errors.VocabError:
            #   logging.warning('deatomizing error')
            #   self.run()
            buf[item].write(atom)
            # Update symmetrical character depths.
            if self.has_symmetrical_tokens:
              _started, depth[item] = GetSymmetricalTokenDepth(atom,
                                                               depth=depth[
                                                                 item])
              started[item] |= _started  # You can't 'unset' the started state.
              running[item] = (
                  not started[item] or (started[item] and depth[item] > 0))
            # Update max length.
            if self.has_max_length and i >= self.max_length:
              running[item] = False
            # Submit sample to processing queue.
            if not running[item]:
              text = buf[item].getvalue()
              self.queue.put(text)
              if logging.get_verbosity() == logging.DEBUG:
                sys.stdout.write(self.sample_header)
                sys.stdout.write(text)
                sys.stdout.flush()
          # Start a new batch if there's nothing left running.
          if not any(running):
            break
          i += 1

      if logging.get_verbosity() == logging.DEBUG:
        sys.stdout.write('\n\n')

  def stop(self) -> None:
    self.stop_signal.set()

  @property
  def stop_requested(self) -> bool:
    return self.stop_signal.isSet()


class SampleConsumer(threading.Thread):
  """Handle generated samples."""

  def __init__(self, producer: SampleProducer, s: samplers.Sampler,
               cache_: labcache.FSCache, q: queue.Queue):
    """Instantiate a SampleConsumer.

    Args:
      producer: A SampleProducer thread.
      s: The host Sampler instance.
      cache_: The sampler cache.
      q: A shared queue.
    """
    super(SampleConsumer, self).__init__()
    self.sample_dir = (cache_.path / 'samples').absolute()
    self.producer = producer
    self.sampler = s
    self.cache = cache_
    self.queue = q
    self.samples = []
    min_samples = self.sampler.config.min_num_samples
    has_min_samples = min_samples >= 0
    # Determine termination criteria.
    if has_min_samples:
      self.term_condition = self.min_samples_cond
      self.max_i = min_samples
      self.progress = self.min_samples_progress
    else:
      self.term_condition = self.null_cond
      self.max_i = progressbar.UnknownLength
      self.progress = self.null_progress

  def min_samples_cond(self) -> bool:
    return len(fs.ls(self.sample_dir)) >= self.max_i

  def null_cond(self) -> bool:
    return False

  def min_samples_progress(self) -> int:
    return min(len(fs.ls(self.sample_dir)), self.max_i)

  def null_progress(self) -> int:
    return len(fs.ls(self.sample_dir))

  def run(self) -> None:
    if not logging.get_verbosity() == logging.DEBUG:
      bar = progressbar.ProgressBar(max_value=self.max_i)
      bar.update(self.progress())

    try:
      while True:
        start_time = labdate.MillisecondsTimestamp()
        # Block while waiting for a new sample to come in.
        sample_text = self.queue.get(timeout=120).strip()
        end_time = labdate.MillisecondsTimestamp()
        sample_id = crypto.sha1_str(sample_text)
        sample = internal_pb2.Sample(text=sample_text,
                                     sample_start_epoch_ms_utc=start_time,
                                     sample_time_ms=end_time - start_time)
        path = self.sample_dir / (sample_id + '.pb')
        pbutil.ToFile(sample, path)
        if self.term_condition != self.null_cond:
          self.samples.append(sample)
        # Update progress bar.
        progress = self.progress()
        if not logging.get_verbosity() == logging.DEBUG:
          bar.update(progress)
        # Determine if we are done sampling.
        if self.term_condition():
          self.producer.stop()
          return
    finally:
      # Always kill the sampler thread.
      print()
      self.producer.stop()


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
