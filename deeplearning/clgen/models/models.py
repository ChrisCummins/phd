"""The CLgen language model."""
import pathlib
import sys
import typing

import humanize
import numpy as np
from absl import flags
from absl import logging

from deeplearning.clgen import cache
from deeplearning.clgen import corpuses
from deeplearning.clgen import errors
from deeplearning.clgen import samplers
from deeplearning.clgen import telemetry
from deeplearning.clgen.models import builders
from deeplearning.clgen.models import data_generators
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.proto import model_pb2
from lib.labm8 import crypto
from lib.labm8 import labdate
from lib.labm8 import lockfile
from lib.labm8 import logutil
from lib.labm8 import pbutil


FLAGS = flags.FLAGS


class Model(object):
  """A CLgen Model.

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
    # Validate config options.
    if config.training.sequence_length < 1:
      raise errors.UserError('TrainingOptions.sequence_length must be >= 1')

    # Attributes that will be lazily set.
    self._model: typing.Optional['keras.models.Sequential'] = None
    self._current_weights_epoch: int = 0

    self.config = model_pb2.Model()
    self.config.CopyFrom(builders.AssertIsBuildable(config))
    self.corpus = corpuses.Corpus(config.corpus)
    self.hash = self._ComputeHash(self.corpus, self.config)
    self.cache = cache.mkcache('model', f'{self.corpus.language}-{self.hash}')
    # Create the necessary cache directories.
    (self.cache.path / 'checkpoints').mkdir(exist_ok=True)
    (self.cache.path / 'samples').mkdir(exist_ok=True)
    (self.cache.path / 'logs').mkdir(exist_ok=True)
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

    Args:
      corpus: A corpus instance.
      config: A Model config proto.

    Returns:
      The unique model ID.
    """
    config_to_hash = model_pb2.Model()
    config_to_hash.CopyFrom(config)
    config_to_hash.ClearField('corpus')
    config_to_hash.training.ClearField('num_epochs')
    return crypto.sha1_list(corpus_.hash,
                            config_to_hash.SerializeToString())

  def GetTrainableModel(self) -> 'keras.models.Sequential':
    """Get the Keras model.

    If there is a cached model description, the model will be initialized from
    that. Else, it is constructed from the proto config.

    Returns:
      A Sequential model instance.
    """
    # Deferred importing of Keras so that we don't have to activate the
    # TensorFlow backend every time we import this module.
    import keras

    if self.cache.get('model.yaml'):
      with open(self.cache['model.yaml']) as f:
        model = keras.models.model_from_yaml(f.read())
    else:
      model = builders.BuildKerasModel(self.config, self.corpus.vocabulary_size)
      with open(self.cache.keypath('model.yaml'), 'w') as f:
        f.write(model.to_yaml())
    model.compile(loss='categorical_crossentropy',
                  optimizer=builders.BuildOptimizer(self.config))
    return model

  @property
  def model(self) -> 'keras.models.Sequential':
    if self._model is None:
      self._model = self.GetTrainableModel()
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
      return self

    epoch_checkpoints = self.epoch_checkpoints
    if len(epoch_checkpoints) >= target_num_epochs:
      # We have already trained a model to at least this number of epochs, so
      # simply the weights from that epoch and call it a day.
      self.model.load_weights(
          epoch_checkpoints[target_num_epochs - 1])
      return self

    # Deferred importing of Keras so that we don't have to activate the
    # TensorFlow backend every time we import this module.
    import keras

    with logutil.TeeLogsToFile('train', self.cache.path / 'logs'):
      if epoch_checkpoints:
        # We have already trained a model at least part of the way to our target
        # number of epochs, so load the most recent one.
        self.model.load_weights(epoch_checkpoints[-1])
        starting_epoch = len(epoch_checkpoints)

      # model.load_weights(self.most_recent_checkpoint_path)
      checkpoint_dir = pathlib.Path(self.cache.keypath('checkpoints'))
      checkpoint_dir.mkdir(parents=True, exist_ok=True)
      file_path = str(
          checkpoint_dir) + "/checkpoint_weights_{epoch:02d}_{loss:.4f}.hdf5"

      callbacks = [
        keras.callbacks.ModelCheckpoint(
            file_path, monitor="loss", verbose=1,
            save_best_only=False, mode="min"),
        telemetry.TrainingLogger(pathlib.Path('/tmp')).KerasCallback(keras),
      ]
      generator = data_generators.AutoGenerator(self.corpus,
                                                self.config.training)
      logging.info('Step counts: %s per epoch, %s left to do, %s total',
                   humanize.intcomma(generator.steps_per_epoch),
                   humanize.intcomma(
                       (target_num_epochs - starting_epoch) *
                       generator.steps_per_epoch),
                   humanize.intcomma(
                       target_num_epochs * generator.steps_per_epoch))
      self.model.fit_generator(generator,
                               steps_per_epoch=generator.steps_per_epoch,
                               epochs=target_num_epochs - starting_epoch,
                               callbacks=callbacks)
      self._current_weights_epoch = self.config.training.num_epochs
      # TODO(cec): Checkpoint callback.
      # stat = self.meta.training_stats.add()
      # stat.batch_num = batch_num + 1
      # stat.time_ms = int(epoch_duration * 1000)
      # stat.training_cost = float(train_cost)
      # self._WriteMetafile()
    return self

  def Train(self) -> 'Model':
    """Train the model.

    Returns:
      The model instance.

    Raises:
      UnableToAcquireLockError: If the model is locked (i.e. there is another
        process currently modifying the model).
    """
    with self.lock.acquire(replace_stale=True):
      return self._LockedTrain()

  def Sample(self, sampler: samplers.Sampler,
             min_num_samples: int) -> typing.List[model_pb2.Sample]:
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
    self.SamplerCache(sampler).mkdir(exist_ok=True)
    self.Train()
    with logutil.TeeLogsToFile(
        f'sampler_{sampler.hash}', self.cache.path / 'logs'):
      logging.info("Sampling: '%s'", sampler.start_text)
      if min_num_samples < 0:
        logging.warning(
            'Entering an infinite sample loop, this process will never end!')
      try:
        encoded_seed = self.corpus.atomizer.AtomizeString(sampler.start_text)
      except errors.VocabError:
        raise errors.InvalidStartText(
            'Sampler start text cannot be encoded using the corpus vocabulary: '
            f"'{sampler.start_text}'")
      # TODO(cec): Update this to use the new sampler API.
      # if sampler.has_symmetrical_tokens:
      #   try:
      #     l = self.corpus.atomizer.AtomizeString(sampler.symmetrical_token_left)
      #     r = self.corpus.atomizer.AtomizeString(
      #         sampler.symmetrical_token_right)
      #     if len(l) > 1 or len(r) > 1:
      #       raise errors.InvalidSymtokTokens(
      #           'Sampler symmetrical depth tokens do not encode to a single '
      #           'token using the corpus vocabulary')
      #   except errors.VocabError:
      #     raise errors.InvalidSymtokTokens(
      #         'Sampler symmetrical depth tokens cannot be encoded using the '
      #         'corpus vocabulary')

      samples = []

      # TODO(cec): Re-implement batched sampling.
      # Use the same vectorizer as the DataGenorator.
      vectorized_seed = np.zeros(
          (1,
           self.config.training.sequence_length, self.corpus.vocabulary_size),
          dtype=np.bool)
      for i, token in enumerate(encoded_seed):
        vectorized_seed[0, i, token] = 1

      # TODO(cec): Add to sampler proto.
      temperature = 1.0
      while True:
        X = np.copy(vectorized_seed)
        sample_in_progress = [sampler.start_text]
        print('=== BEGIN CLGEN SAMPLE ===', sampler.start_text, sep='\n',
              end='')
        start_time = labdate.MillisecondsTimestamp()
        # Save a few cycles by not going via the model() property every time we
        # need to access it.
        model = self.model
        while True:
          predictions = model.predict(X, verbose=0)[0]
          next_index = WeightedPick(predictions, temperature)
          token = self.corpus.atomizer.decoder[next_index]
          sys.stdout.write(token)
          sample_in_progress.append(token)
          activations = np.zeros((1, 1, self.corpus.vocabulary_size),
                                 dtype=np.bool)
          activations[0, 0, next_index] = 1
          X = np.concatenate((X[:, 1:, :], activations), axis=1)
          if sampler.SampleIsComplete(sample_in_progress):
            break
        end_time = labdate.MillisecondsTimestamp()
        sample = model_pb2.Sample(text=''.join(sample_in_progress),
                                  sample_start_epoch_ms_utc=start_time,
                                  sample_time_ms=end_time - start_time,
                                  num_tokens=len(sample_in_progress))
        sample_id = crypto.sha256_str(sample.text)
        p = self.SamplerCache(sampler) / f'{sample_id}.pbtxt'
        pbutil.ToFile(sample, p)
        sys.stdout.write('\n')
        if min_num_samples > 0:
          samples.append(sample)
          if len(samples) >= min_num_samples:
            break

    return samples

  def SamplerCache(self, sampler: samplers.Sampler) -> pathlib.Path:
    """Get the path to a sampler cache.

    Args:
      sampler: A Sampler instance.

    Returns:
      A path to a directory. Note that this directory may not exist - it is
      created only after a call to Sample().
    """
    return self.cache.path / 'samples' / sampler.hash

  def _WriteMetafile(self) -> None:
    pbutil.ToFile(self.meta, pathlib.Path(self.cache.keypath('META.pbtxt')))

  @property
  def is_trained(self) -> bool:
    """Return whether the model has previously been trained."""
    return len(self.epoch_checkpoints) >= self.config.training.num_epochs

  @property
  def lock(self) -> lockfile.LockFile:
    """Get the lockfile."""
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


def WeightedPick(predictions: np.ndarray, temperature: float) -> np.ndarray:
  """Make a weighted choice from a predictions array."""
  predictions = np.log(np.asarray(predictions).astype('float64')) / temperature
  predictions_exp = np.exp(predictions)
  predictions = predictions_exp / np.sum(predictions_exp)
  predictions = np.random.multinomial(1, predictions, 1)
  return np.argmax(predictions)
