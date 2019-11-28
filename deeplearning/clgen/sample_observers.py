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
"""This file contains the SampleObserver interface and concrete subclasses."""
import pathlib

from deeplearning.clgen.proto import model_pb2
from labm8.py import app
from labm8.py import crypto
from labm8.py import fs
from labm8.py import pbutil

FLAGS = app.FLAGS


class SampleObserver(object):
  """An observer that is notified when new samples are produced.

  During sampling of a model, sample observers are notified for each new
  sample produced. Additionally, sample observers determine when to terminate
  sampling.
  """

  def Specialize(self, model, sampler) -> None:
    """Specialize the sample observer to a model and sampler combination.

    This enables the observer to set state specialized to a specific model and
    sampler. This is guaranteed to be called before OnSample(), and
    sets that the model and sampler for each subsequent call to OnSample(),
    until the next call to Specialize().

    Subclasses do not need to override this method.

    Args:
      model: The model that is being sampled.
      sampler: The sampler that is being used.
    """
    pass

  def OnSample(self, sample: model_pb2.Sample) -> bool:
    """Sample notification callback.

    Args:
      sample: The newly created sample message.

    Returns:
      True if sampling should continue, else False. Batching of samples means
      that returning False does not guarantee that sampling will terminate
      immediately, and OnSample() may be called again.
    """
    raise NotImplementedError("abstract class")


class MaxSampleCountObserver(SampleObserver):
  """An observer that terminates sampling after a finite number of samples."""

  def __init__(self, min_sample_count: int):
    if min_sample_count <= 0:
      raise ValueError(
          f"min_sample_count must be >= 1. Received: {min_sample_count}")

    self._sample_count = 0
    self._min_sample_count = min_sample_count

  def OnSample(self, sample: model_pb2.Sample) -> bool:
    """Sample receive callback. Returns True if sampling should continue."""
    self._sample_count += 1
    return self._sample_count < self._min_sample_count


class SaveSampleTextObserver(SampleObserver):
  """An observer that creates a file of the sample text for each sample."""

  def __init__(self, path: pathlib.Path):
    self.path = pathlib.Path(path)
    self.path.mkdir(parents=True, exist_ok=True)

  def OnSample(self, sample: model_pb2.Sample) -> bool:
    """Sample receive callback. Returns True if sampling should continue."""
    sample_id = crypto.sha256_str(sample.text)
    path = self.path / f'{sample_id}.txt'
    fs.Write(path, sample.text.encode('utf-8'))
    return True


class PrintSampleObserver(SampleObserver):
  """An observer that prints the text of each sample that is generated."""

  def OnSample(self, sample: model_pb2.Sample) -> bool:
    """Sample receive callback. Returns True if sampling should continue."""
    print(f'=== CLGEN SAMPLE ===\n\n{sample.text}\n')
    return True


class InMemorySampleSaver(SampleObserver):
  """An observer that saves all samples in-memory."""

  def __init__(self):
    self.samples = []

  def OnSample(self, sample: model_pb2.Sample) -> bool:
    """Sample receive callback. Returns True if sampling should continue."""
    self.samples.append(sample)
    return True


class LegacySampleCacheObserver(SampleObserver):
  """Backwards compatability implementation of the old sample caching behavior.

  In previous versions of CLgen, model sampling would silently (and always)
  create sample protobufs in the sampler cache, located at:

    CLGEN_CACHE/models/MODEL/samples/SAMPLER

  This sample observer provides equivalent behavior.
  """

  def __init__(self):
    self.cache_path = None

  def Specialize(self, model, sampler) -> None:
    """Specialize observer to a model and sampler combination."""
    self.cache_path = model.SamplerCache(sampler)
    self.cache_path.mkdir(exist_ok=True)

  def OnSample(self, sample: model_pb2.Sample) -> bool:
    """Sample receive callback. Returns True if sampling should continue."""
    sample_id = crypto.sha256_str(sample.text)
    sample_path = self.cache_path / f'{sample_id}.pbtxt'
    pbutil.ToFile(sample, sample_path)
    return True
