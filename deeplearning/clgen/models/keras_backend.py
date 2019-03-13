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
import io
import pathlib
import typing

import numpy as np

from deeplearning.clgen import samplers
from deeplearning.clgen import telemetry
from deeplearning.clgen.models import backends
from deeplearning.clgen.models import builders
from deeplearning.clgen.models import data_generators
from labm8 import app
from labm8 import humanize
from labm8 import logutil

FLAGS = app.FLAGS


class KerasBackend(backends.BackendBase):
  """A model with an embedding layer, using a keras backend."""

  def __init__(self, *args, **kwargs):
    """Instantiate a model.

    Args:
      args: Arguments to be passed to BackendBase.__init__().
      kwargs: Arguments to be passed to BackendBase.__init__().
    """
    super(KerasBackend, self).__init__(*args, **kwargs)

    # Create the necessary cache directories.
    (self.cache.path / 'embeddings').mkdir(exist_ok=True)

    # Attributes that will be lazily set.
    self._training_model: typing.Optional['keras.models.Sequential'] = None
    self._inference_model: typing.Optional['keras.models.Sequential'] = None
    self._inference_batch_size: typing.Optional[int] = None

    self.inference_indices = None
    self.inference_model = None

  def GetTrainingModel(self) -> 'keras.models.Sequential':
    """Get the Keras model."""
    if self._training_model:
      return self._training_model
    self._training_model = self.Train()
    return self._training_model

  def Train(self, corpus) -> 'keras.models.Sequential':
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
    model = builders.BuildKerasModel(self.config, self.atomizer.vocab_size)
    with open(self.cache.keypath('model.yaml'), 'w') as f:
      f.write(model.to_yaml())
    model.compile(
        loss='categorical_crossentropy',
        optimizer=builders.BuildOptimizer(self.config))

    # Print a model summary.
    buf = io.StringIO()
    model.summary(print_fn=lambda x: buf.write(x + '\n'))
    app.Log(1, 'Model summary:\n%s', buf.getvalue())

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
        for _, token in sorted(
            self.atomizer.decoder.items(), key=lambda x: x[0]):
          f.write(Escape(token) + '\n')

    target_num_epochs = self.config.training.num_epochs
    starting_epoch = 0

    epoch_checkpoints = self.epoch_checkpoints
    if len(epoch_checkpoints) >= target_num_epochs:
      # We have already trained a model to at least this number of epochs, so
      # simply the weights from that epoch and call it a day.
      app.Log(1, 'Loading weights from %s',
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
        app.Log(1, 'Resuming training from epoch %d.', starting_epoch)
        model.load_weights(epoch_checkpoints[-1])

      callbacks = [
          keras.callbacks.ModelCheckpoint(
              str(self.cache.path / 'checkpoints' / '{epoch:03d}.hdf5'),
              verbose=1,
              mode="min",
              save_best_only=False),
          keras.callbacks.TensorBoard(
              str(self.cache.path / 'embeddings'),
              write_graph=True,
              embeddings_freq=1,
              embeddings_metadata={
                  'embedding_1':
                  str(self.cache.path / 'embeddings' / 'metadata.tsv'),
              }),
          telemetry.TrainingLogger(
              self.cache.path / 'logs').KerasCallback(keras),
      ]

      generator = data_generators.AutoGenerator(corpus, self.config.training)
      steps_per_epoch = (corpus.encoded.token_count - 1) // (
          self.config.training.batch_size *
          self.config.training.sequence_length)
      app.Log(
          1, 'Step counts: %s per epoch, %s left to do, %s total',
          humanize.Commas(steps_per_epoch),
          humanize.Commas(
              (target_num_epochs - starting_epoch) * steps_per_epoch),
          humanize.Commas(target_num_epochs * steps_per_epoch))
      model.fit_generator(
          generator,
          steps_per_epoch=steps_per_epoch,
          callbacks=callbacks,
          initial_epoch=starting_epoch,
          epochs=target_num_epochs)
    return model

  def GetInferenceModel(self) -> 'keras.models.Sequential':
    """Like training model, but with different batch size."""
    if self._inference_model:
      return self._inference_model

    # Deferred importing of Keras so that we don't have to activate the
    # TensorFlow backend every time we import this module.
    import keras

    app.Log(1, 'Building inference model.')
    model = self.GetTrainingModel()
    config = model.get_config()
    app.Log(1, 'Sampling with batch size %d', sampler.batch_size)
    config[0]['config']['batch_input_shape'] = (sampler.batch_size, 1)
    inference_model = keras.models.Sequential.from_config(config)
    inference_model.trainable = False
    inference_model.set_weights(model.get_weights())
    self._inference_model = inference_model
    self._inference_batch_size = sampler.batch_size
    return inference_model

  def InitSampling(self,
                   sampler: samplers.Sampler,
                   seed: typing.Optional[int] = None) -> None:
    self.inference_model = self.GetInferenceModel()
    if seed is not None:
      np.random.seed(seed)

  def InitSampleBatch(self, sampler: samplers.Sampler) -> None:
    self.inference_model.reset_states()
    # Set internal states from seed text.
    for index in sampler.encoded_start_text[:-1]:
      x = np.array([[index]] * sampler.batch_size)
      # input shape: (batch_size, 1)
      self.inference_model.predict(x)

    self.inference_indices = (
        [sampler.encoded_start_text[-1]] * sampler.batch_size)

  def SampleNextIndices(self, sampler: samplers.Sampler, done: np.ndarray):
    del done
    result = np.zeros((sampler.batch_size, 1024))
    for idx in range(1024):
      # Predict the next index for the entire batch.
      x = np.reshape(self.inference_indices, [sampler.batch_size, 1])
      # Input shape: (batch_size, 1).
      probabilities = self.inference_model.predict(x)
      # Output shape: (batch_size, 1, vocab_size).
      self.inference_indices = [
          WeightedPick(p.squeeze(), sampler.temperature) for p in probabilities
      ]
      result[:, idx] = self.inference_indices
    return result

  def InferenceManifest(self) -> typing.List[pathlib.Path]:
    """Return the list of files which are required for model inference.

    Returns:
      A list of absolute paths.
    """
    raise NotImplementedError

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
    return [
        checkpoint_dir / x
        for x in sorted(pathlib.Path(self.cache['checkpoints']).iterdir())
    ]

  @property
  def is_trained(self) -> bool:
    """Return whether the model has previously been trained."""
    return len(self.epoch_checkpoints) >= self.config.training.num_epochs


def WeightedPick(predictions: np.ndarray, temperature: float) -> int:
  """Make a weighted choice from a predictions array."""
  predictions = np.log(np.asarray(predictions).astype('float64')) / temperature
  predictions_exp = np.exp(predictions)
  # Normalize the probabilities.
  predictions = predictions_exp / np.sum(predictions_exp)
  predictions = np.random.multinomial(1, predictions, 1)
  return np.argmax(predictions)
