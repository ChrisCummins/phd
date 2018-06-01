"""The CLgen language model."""
import os
import pathlib
import typing

import humanize
import numpy as np
import sys
from absl import flags
from absl import logging

from deeplearning.clgen import cache
from deeplearning.clgen import errors
from deeplearning.clgen import samplers
from deeplearning.clgen import telemetry
from deeplearning.clgen.corpuses import corpuses
from deeplearning.clgen.models import builders
from deeplearning.clgen.models import data_generators
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import telemetry_pb2
from lib.labm8 import crypto
from lib.labm8 import labdate
from lib.labm8 import lockfile
from lib.labm8 import logutil
from lib.labm8 import pbutil


FLAGS = flags.FLAGS


def OneHotEncode(indices: np.ndarray, vocabulary_size: int):
  """One-hot encode a sequence of encoded token indices.

    Args:
      indices: A 1D array of vocabulary indices.
      vocabulary_size: The size of the vocabulary.

    Returns:
      A 2D array of one-hot encoded sequences.
    """
  return np.eye(vocabulary_size)[indices]


def SampleProbabilities(probs, clip_after=10):
  probs = np.array(probs, dtype=np.float64)
  # Set all probabilities after clip_after to 0.
  probs[np.argsort(probs)[:-clip_after]] = 0
  probs /= np.sum(probs)
  sampled_index = np.random.choice(len(probs), p=probs)
  return sampled_index


def GetInferenceModel(model, batch_size=1, seq_len=1):
  """Like training model, but with batch size 1."""
  import keras
  logging.info("building inference model.")
  config = model.get_config()
  # edit batch_size and seq_len
  config[0]["config"]["batch_input_shape"] = (batch_size, seq_len)
  inference_model = keras.models.Sequential.from_config(config)
  inference_model.trainable = False
  return inference_model


def BatchGenerator(sequence, vocabulary_size, batch_size=64, seq_len=64,
                   one_hot_features=False, one_hot_labels=False):
  num_batches = (len(sequence) - 1) // (batch_size * seq_len)
  if num_batches == 0:
    raise ValueError(
        "No batches created. Use smaller batch size or sequence length.")
  logging.info("number of batches: %s.", num_batches)
  rounded_len = num_batches * batch_size * seq_len
  logging.info("effective text length: %s.", rounded_len)

  x = np.reshape(sequence[: rounded_len], [batch_size, num_batches * seq_len])
  if one_hot_features:
    x = OneHotEncode(x, vocabulary_size)
  logging.info("x shape: %s.", x.shape)

  y = np.reshape(sequence[1: rounded_len + 1],
                 [batch_size, num_batches * seq_len])
  if one_hot_labels:
    y = OneHotEncode(y, vocabulary_size)
  logging.info("y shape: %s.", y.shape)

  epoch = 0
  while True:
    # roll so that no need to reset rnn states over epochs
    x_epoch = np.split(np.roll(x, -epoch, axis=0), num_batches, axis=1)
    y_epoch = np.split(np.roll(y, -epoch, axis=0), num_batches, axis=1)
    for batch in range(num_batches):
      yield data_generators.DataBatch(X=x_epoch[batch], y=y_epoch[batch])
    epoch += 1


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
    self.cache = cache.mkcache('model', self.hash)
    # Create the necessary cache directories.
    (self.cache.path / 'checkpoints').mkdir(exist_ok=True)
    (self.cache.path / 'samples').mkdir(exist_ok=True)
    (self.cache.path / 'logs').mkdir(exist_ok=True)

    # Create symlink to encoded corpus.
    symlink = self.cache.path / 'corpus'
    if not symlink.is_symlink():
      os.symlink(self.corpus.encoded.database_path.parent, symlink)

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
    # TODO(cec): Capture using StringIO as print_fn and log:
    model.summary()
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
      logging.info('Loading weights from %s',
                   epoch_checkpoints[target_num_epochs - 1])
      self.model.load_weights(
          epoch_checkpoints[target_num_epochs - 1])
      return self

    logging.info('Training model for %d of %d epochs',
                 target_num_epochs - starting_epoch, target_num_epochs)

    # Deferred importing of Keras so that we don't have to activate the
    # TensorFlow backend every time we import this module.
    import keras

    with logutil.TeeLogsToFile('train', self.cache.path / 'logs'):
      if epoch_checkpoints:
        # We have already trained a model at least part of the way to our target
        # number of epochs, so load the most recent one.
        logging.info('Loading weights from %s',
                     epoch_checkpoints[target_num_epochs - 1])
        self.model.load_weights(epoch_checkpoints[-1])
        starting_epoch = len(epoch_checkpoints)

      # model.load_weights(self.most_recent_checkpoint_path)
      checkpoint_dir = pathlib.Path(self.cache.keypath('checkpoints'))
      checkpoint_dir.mkdir(parents=True, exist_ok=True)

      callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(checkpoint_dir / "{epoch:03d}.hdf5"), verbose=1, mode="min",
            save_best_only=False),
        # keras.callbacks.TensorBoard(
        #     '${EMBEDDINGS_DIR}', write_graph=True, embeddings_freq=1,
        #     embeddings_metadata={"embedding_1": "PATH_TO_EMBEDDINGS"}),
        telemetry.TrainingLogger(self.cache.path / 'logs').KerasCallback(keras),
      ]
      generator = data_generators.AutoGenerator(
          self.corpus, self.config.training)

      generator_ = BatchGenerator(generator.encoded_corpus,
                                  self.corpus.atomizer.vocab_size,
                                  batch_size=self.config.training.batch_size,
                                  seq_len=self.config.training.sequence_length,
                                  one_hot_features=False, one_hot_labels=True)

      logging.info('Step counts: %s per epoch, %s left to do, %s total',
                   humanize.intcomma(generator.steps_per_epoch),
                   humanize.intcomma(
                       (target_num_epochs - starting_epoch) *
                       generator.steps_per_epoch),
                   humanize.intcomma(
                       target_num_epochs * generator.steps_per_epoch))
      self.model.fit_generator(generator_,
                               steps_per_epoch=generator.steps_per_epoch,
                               epochs=target_num_epochs - starting_epoch,
                               callbacks=callbacks)
      self._current_weights_epoch = self.config.training.num_epochs
    return self

  def Train(self) -> 'Model':
    """Train the model.

    Returns:
      The model instance.

    Raises:
      UnableToAcquireLockError: If the model is locked (i.e. there is another
        process currently modifying the model).
    """
    self.corpus.Create()
    with self.lock.acquire(replace_stale=True, block=True):
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

      sampler.Specialize(self.corpus.atomizer)
      # TODO(cec): Re-implement batched sampling.

      # Save a few cycles by not going via the model() property every time we
      # need to access it.
      model = GetInferenceModel(self.model)

      samples = []
      while True:
        print('\n=== BEGIN CLGEN SAMPLE ===\n', sampler.start_text, sep='\n',
              end='')

        model.reset_states()
        sample_in_progress = sampler.tokenized_start_text.copy()
        start_time = labdate.MillisecondsTimestamp()

        # Set internal states from seed text.
        for index in sampler.encoded_start_text[:-1]:
          x = np.array([[index]])
          # input shape: (1, 1)
          model.predict(x)

        next_index = sampler.encoded_start_text[-1]
        while True:
          x = np.array([[next_index]])
          # Input shape: (1, 1).
          probabilities = model.predict(x)
          # Output shape: (1, 1, vocab_size).
          # TODO(cec): Make configurable, or use old weighted pick.
          top_n = 10
          next_index = SampleProbabilities(probabilities.squeeze(), top_n)
          # append to sequence
          token = self.corpus.atomizer.decoder[next_index]
          sample_in_progress.append(token)
          sys.stdout.write(token)
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

  def TrainingTelemetry(self) -> typing.List[telemetry_pb2.ModelEpochTelemetry]:
    """Get the training telemetry data."""
    return telemetry.TrainingLogger(self.cache.path / 'logs').EpochTelemetry()

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
