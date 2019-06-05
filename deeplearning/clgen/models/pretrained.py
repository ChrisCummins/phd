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
"""This file defines the PreTrainedModel class."""
import pathlib
import typing

import numpy as np

from deeplearning.clgen import errors
from deeplearning.clgen import sample_observers as sample_observers_lib
from deeplearning.clgen import samplers
from deeplearning.clgen import telemetry
from deeplearning.clgen.corpuses import atomizers
from deeplearning.clgen.models import keras_backend
from deeplearning.clgen.models import tensorflow_backend
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import telemetry_pb2
from labm8 import app
from labm8 import cache
from labm8 import humanize
from labm8 import labdate
from labm8 import pbutil

FLAGS = app.FLAGS


class PreTrainedModel(object):
  """A pre-trained model is a model which can be used only for inference."""

  def __init__(self, path: pathlib.Path):
    self.path = path.absolute()
    self.cache = cache.FSCache(self.path)
    self.corpus = NullCorpus()
    self.config = pbutil.FromFile(self.path / 'META.pbtxt',
                                  internal_pb2.ModelMeta()).config
    self.atomizer = atomizers.AtomizerBase.FromFile(self.path / 'atomizer')
    self.backend = {
        model_pb2.NetworkArchitecture.TENSORFLOW:
        tensorflow_backend.TensorFlowBackend,
        model_pb2.NetworkArchitecture.KERAS: keras_backend.KerasBackend,
    }[self.config.architecture.backend](self.config, self.cache, self.atomizer)

  def Train(self):
    """The training process for a pre-trained model is a no-op."""
    pass

  def TrainingTelemetry(self) -> typing.List[telemetry_pb2.ModelEpochTelemetry]:
    """Get the training telemetry data."""
    return telemetry.TrainingLogger(self.cache.path / 'logs').EpochTelemetry()

  def _SampleBatch(
      self, sampler: samplers.Sampler, atomizer: atomizers.AtomizerBase,
      sample_observers: typing.List[sample_observers_lib.SampleObserver]
  ) -> typing.List[model_pb2.Sample]:
    """Run a single iteration of the batched sample inner-loop."""
    samples_in_progress = [
        sampler.tokenized_start_text.copy() for _ in range(sampler.batch_size)
    ]
    done = np.zeros(sampler.batch_size, dtype=np.bool)
    start_time = labdate.MillisecondsTimestamp()
    wall_time_start = start_time

    self.backend.InitSampleBatch(sampler)

    # The return value of this method. If any of the sample_observers return
    # False, this value is set to False.
    continue_sampling = True

    # Sampling loop. Continues until all samples in the batch are done.
    while not done.all():
      indices = self.backend.SampleNextIndices(sampler, done)

      # Iterate over all samples in batch to determine whether they're
      # done.
      for i in range(sampler.batch_size):
        if done[i]:
          continue

        for index in indices[i]:
          samples_in_progress[i].append(atomizer.decoder[index])
          if sampler.SampleIsComplete(samples_in_progress[i]):
            end_time = labdate.MillisecondsTimestamp()
            done[i] = 1
            sample = model_pb2.Sample(text=''.join(samples_in_progress[i]),
                                      sample_start_epoch_ms_utc=start_time,
                                      sample_time_ms=end_time - start_time,
                                      wall_time_ms=end_time - wall_time_start,
                                      num_tokens=len(samples_in_progress[i]))
            # Notify sample observers.
            continue_sampling &= all(
                [not obs.OnSample(sample) for obs in sample_observers])

            # Wall sample time is the difference between the end of the previous
            # sample and the end of the current sample.
            wall_time_start = labdate.MillisecondsTimestamp()
            break

    return continue_sampling

  def Sample(self,
             sampler: samplers.Sampler,
             sample_observers: typing.List[sample_observers_lib.SampleObserver],
             seed: int = None) -> None:
    """Sample a model.

    This method uses the observer model, returning nothing. To access the
    samples produced, implement a SampleObserver and pass it in as an argument.
    Sampling continues indefinitely until one of the sample observers returns
    False when notified of a new sample.

    If the model is not already trained, calling Sample() first trains the
    model. Thus a call to Sample() is equivalent to calling Train() then
    Sample().

    Args:
      sampler: The sampler to sample using.
      sample_observers: A list of SampleObserver objects that are notified of
        new generated samples.
      seed: A numeric value to seed the RNG with. If not present, the RNG is
        seeded randomly.

    Raises:
      UserError: If called with no sample observers.
      UnableToAcquireLockError: If the model is locked (i.e. there is another
        process currently modifying the model).
      InvalidStartText: If the sampler start text cannot be encoded.
      InvalidSymtokTokens: If the sampler symmetrical depth tokens cannot be
        encoded.
    """
    if not sample_observers:
      raise errors.UserError("Cannot sample without any observers")

    sample_start_time = labdate.MillisecondsTimestamp()
    app.Log(1, "Sampling: '%s'", sampler.start_text)

    atomizer = self.atomizer
    sampler.Specialize(atomizer)
    self.backend.InitSampling(sampler, seed)
    [obs.Specialize(self, sampler) for obs in sample_observers]

    batch_count = 0
    while self._SampleBatch(sampler, atomizer, sample_observers):
      batch_count += 1

    time_now = labdate.MillisecondsTimestamp()
    app.Log(
        1, 'Produced %s sample batches at a rate of %s ms / batch.',
        humanize.Commas(batch_count),
        humanize.Commas(
            int((time_now - sample_start_time) / max(batch_count, 1))))

  def SamplerCache(self, sampler: samplers.Sampler) -> pathlib.Path:
    """Get the path to a sampler cache.

    Args:
      sampler: A Sampler instance.

    Returns:
      A path to a directory. Note that this directory may not exist - it is
      created only after a call to Sample().
    """
    return self.cache.path / 'samples' / sampler.hash


class NullCorpus(object):
  """Corpus for a pre-trained model."""

  def Create(self):
    """The creation process for a null corpus is a no-op."""
    pass
