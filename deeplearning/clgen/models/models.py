"""The CLgen language model."""
import io
import os
import pathlib
import typing

import humanize
import numpy as np
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

flags.DEFINE_bool(
    'experimental_batched_sampling', False,
    'Enable an experimental batched sampling feature. THIS FEATURE IS STILL '
    'EXPERIMENTAL AND HAS NOT BEEN THOROUGHLY REVIEWED OR UNDERSTOOD.')


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
    self._training_model: typing.Optional['keras.models.Sequential'] = None
    self._inference_model: typing.Optional['keras.models.Sequential'] = None

    self.config = model_pb2.Model()
    self.config.CopyFrom(builders.AssertIsBuildable(config))
    self.corpus = corpuses.Corpus(config.corpus)
    self.hash = self._ComputeHash(self.corpus, self.config)
    self.cache = cache.mkcache('model', self.hash)
    # Create the necessary cache directories.
    (self.cache.path / 'checkpoints').mkdir(exist_ok=True)
    (self.cache.path / 'embeddings').mkdir(exist_ok=True)
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

  def GetTrainingModel(self) -> 'keras.models.Sequential':
    """Get the Keras model."""
    if self._training_model:
      return self._training_model
    self.corpus.Create()
    self._training_model = self.GetTrainedModel()
    return self._training_model

  def GetInferenceModel(self) -> 'keras.models.Sequential':
    """Like training model, but with batch size 1."""
    if self._inference_model:
      return self._inference_model

    import keras

    # Deferred importing of Keras so that we don't have to activate the
    # TensorFlow backend every time we import this module.
    logging.info('Building inference model.')
    model = self.GetTrainingModel()
    config = model.get_config()
    # TODO(cec): Decide on whether this should be on by default, or in the
    # sampler.proto.
    if FLAGS.experimental_batched_sampling:
      # Read the embedding output size.
      batch_size = config[0]['config']['output_dim']
    else:
      batch_size = 1
    logging.info('Sampling with batch size %d', batch_size)
    config[0]['config']['batch_input_shape'] = (batch_size, 1)
    inference_model = keras.models.Sequential.from_config(config)
    inference_model.trainable = False
    inference_model.set_weights(model.get_weights())
    self._inference_model = inference_model
    return inference_model, batch_size

  def GetTrainedModel(self) -> 'keras.models.Sequential':
    """Get and return a trained Keras model."""
    with self.lock.acquire(replace_stale=True, block=True):
      model = self._LockedTrain()
    total_time_ms = sum(
        t.epoch_wall_time_ms
        for t in self.TrainingTelemetry()[:self.config.training.num_epochs])
    logging.info('Trained model for %d epochs in %s ms (%s).',
                 self.config.training.num_epochs,
                 humanize.intcomma(total_time_ms),
                 humanize.naturaldelta(total_time_ms / 1000))
    return model

  def _LockedTrain(self) -> 'keras.models.Sequential':
    """Locked training.

    If there are cached epoch checkpoints, the one closest to the target number
    of epochs will be loaded, and the model will be trained for only the
    remaining number of epochs, if any. This means that calling this function
    twice will only actually train the model the first time, and all subsequent
    calls will be no-ops.

    This method must only be called when the model is locked.

    Returns:
      The trained Keras model.
    """
    model = builders.BuildKerasModel(self.config, self.corpus.vocab_size)
    with open(self.cache.keypath('model.yaml'), 'w') as f:
      f.write(model.to_yaml())
    model.compile(loss='categorical_crossentropy',
                  optimizer=builders.BuildOptimizer(self.config))

    # Print a model summary.
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + '\n'))
    logging.info('Model summary:\n%s', buf.getvalue())

    # TODO(cec): Add an atomizer.CreateVocabularyFile() method, with frequency
    # counts for a given corpus.
    def Escape(token: str) -> str:
      """Make a token visible and printable."""
      if token == '\t':
        return '\\t'
      elif token == '\n':
        return '\\n'
      elif not token.strip():
        return f"'{token}'"
      else:
        return token

    if not (self.cache.path / 'embeddings' / 'metadata.tsv').is_file():
      with open(self.cache.path / 'embeddings' / 'metadata.tsv', 'w') as f:
        for _, token in sorted(self.corpus.atomizer.decoder.items(),
                               key=lambda x: x[0]):
          f.write(Escape(token) + '\n')

    target_num_epochs = self.config.training.num_epochs
    starting_epoch = 0

    epoch_checkpoints = self.epoch_checkpoints
    if len(epoch_checkpoints) >= target_num_epochs:
      # We have already trained a model to at least this number of epochs, so
      # simply the weights from that epoch and call it a day.
      logging.info('Loading weights from %s',
                   epoch_checkpoints[target_num_epochs - 1])
      model.load_weights(epoch_checkpoints[target_num_epochs - 1])
      return model

    # Now entering the point at which training is inevitable.
    with logutil.TeeLogsToFile('train', self.cache.path / 'logs'):
      # Deferred importing of Keras so that we don't have to activate the
      # TensorFlow backend every time we import this module.
      import keras

      if epoch_checkpoints:
        # We have already trained a model at least part of the way to our target
        # number of epochs, so load the most recent one.
        starting_epoch = len(epoch_checkpoints)
        logging.info('Resuming training from epoch %d.', starting_epoch)
        model.load_weights(epoch_checkpoints[-1])

      callbacks = [
        keras.callbacks.ModelCheckpoint(
            str(self.cache.path / 'checkpoints' / '{epoch:03d}.hdf5'),
            verbose=1, mode="min", save_best_only=False),
        keras.callbacks.TensorBoard(
            str(self.cache.path / 'embeddings'), write_graph=True,
            embeddings_freq=1, embeddings_metadata={
              'embedding_1': str(
                  self.cache.path / 'embeddings' / 'metadata.tsv'),
            }),
        telemetry.TrainingLogger(self.cache.path / 'logs').KerasCallback(keras),
      ]

      generator = data_generators.AutoGenerator(
          self.corpus, self.config.training)
      steps_per_epoch = (self.corpus.encoded.token_count - 1) // (
          self.config.training.batch_size *
          self.config.training.sequence_length)
      logging.info('Step counts: %s per epoch, %s left to do, %s total',
                   humanize.intcomma(steps_per_epoch),
                   humanize.intcomma((target_num_epochs - starting_epoch) *
                                     steps_per_epoch),
                   humanize.intcomma(target_num_epochs * steps_per_epoch))
      model.fit_generator(
          generator, steps_per_epoch=steps_per_epoch, callbacks=callbacks,
          initial_epoch=starting_epoch, epochs=target_num_epochs)
    return model

  def Train(self) -> 'Model':
    """Train the model.

    Returns:
      The model instance.

    Raises:
      UnableToAcquireLockError: If the model is locked (i.e. there is another
        process currently modifying the model).
    """
    self.corpus.Create()
    self.GetTrainedModel()
    return self

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
    sample_count = 1
    self.SamplerCache(sampler).mkdir(exist_ok=True)
    model, batch_size = self.GetInferenceModel()
    with logutil.TeeLogsToFile(
        f'sampler_{sampler.hash}', self.cache.path / 'logs'):
      logging.info("Sampling: '%s'", sampler.start_text)
      if min_num_samples < 0:
        logging.warning(
            'Entering an infinite sample loop, this process will never end!')
      sample_start_time = labdate.MillisecondsTimestamp()

      sampler.Specialize(self.corpus.atomizer)
      samples = []
      while True:
        model.reset_states()
        samples_in_progress = [
          sampler.tokenized_start_text.copy()
          for _ in range(batch_size)]
        start_time = labdate.MillisecondsTimestamp()
        wall_time_start = start_time

        # Set internal states from seed text.
        for index in sampler.encoded_start_text[:-1]:
          x = np.array([[index]] * batch_size)
          # input shape: (batch_size, 1)
          model.predict(x)

        next_index = sampler.encoded_start_text[-1]
        done = np.zeros(batch_size)
        while True:
          # Predict the next index for the entire batch.
          x = np.array([[next_index]] * batch_size)
          # Input shape: (bath_size, 1).
          probabilities = model.predict(x)
          # Output shape: (batch_size, 1, vocab_size).
          next_indices = [
            WeightedPick(p.squeeze(), sampler.temperature)
            for p in probabilities
          ]
          # Append to sequences.
          for i, next_index in enumerate(next_indices):
            if done[i]:
              continue

            token = self.corpus.atomizer.decoder[next_index]
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

          if done.all():
            break

        if len(samples) >= min_num_samples:
          now = labdate.MillisecondsTimestamp()
          logging.info(
              'Produced %s samples at a rate of %s ms / sample.',
              humanize.intcomma(len(samples)),
              humanize.intcomma(int((now - sample_start_time) / len(samples))))
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


def WeightedPick(predictions: np.ndarray, temperature: float) -> int:
  """Make a weighted choice from a predictions array."""
  predictions = np.log(np.asarray(predictions).astype('float64')) / temperature
  predictions_exp = np.exp(predictions)
  # Normalize the probabilities.
  predictions = predictions_exp / np.sum(predictions_exp)
  predictions = np.random.multinomial(1, predictions, 1)
  return np.argmax(predictions)
