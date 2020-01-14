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
"""Unit tests for //deeplearning/clgen/models/keras_backend.py."""
import checksumdir
import numpy as np

from deeplearning.clgen import sample_observers
from deeplearning.clgen.models import keras_backend
from deeplearning.clgen.models import models
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import telemetry_pb2
from labm8.py import app
from labm8.py import crypto
from labm8.py import pbutil
from labm8.py import test

FLAGS = app.FLAGS

pytest_plugins = ["deeplearning.clgen.tests.fixtures"]


class MockSampler(object):
  """Mock class for a Sampler."""

  # The default value for start_text has been chosen to only use characters and
  # words from the abc_corpus, so that it may be encoded using the vocabulary
  # of that corpus.
  def __init__(
    self, start_text: str = "H", hash: str = "hash", batch_size: int = 1
  ):
    self.start_text = start_text
    self.encoded_start_text = np.array([1, 2, 3])
    self.tokenized_start_text = ["a", "b", "c"]
    self.temperature = 1.0
    self.hash = hash
    self.batch_size = batch_size

  @staticmethod
  def Specialize(atomizer):
    """Atomizer.Specialize() mock."""
    pass

  @staticmethod
  def SampleIsComplete(sample_in_progress):
    """Crude 'maxlen' mock."""
    return len(sample_in_progress) >= 10


@test.Fixture(scope="function")
def abc_keras_model_config(abc_model_config: model_pb2.Model):
  """A test fixture for a simple model with a Keras backend."""
  abc_model_config.architecture.backend = model_pb2.NetworkArchitecture.KERAS
  return abc_model_config


# KerasBackend tests.


def test_KerasBackend_directories(clgen_cache_dir, abc_keras_model_config):
  """A newly instantiated model's cache has checkpoint and sample dirs."""
  del clgen_cache_dir
  m = models.Model(abc_keras_model_config)
  assert (m.cache.path / "embeddings").is_dir()
  assert not list((m.cache.path / "embeddings").iterdir())


def test_KerasBackend_epoch_checkpoints_untrained(
  clgen_cache_dir, abc_keras_model_config
):
  """Test that an untrained model has no checkpoint files."""
  del clgen_cache_dir
  m = models.Model(abc_keras_model_config)
  assert not m.backend.epoch_checkpoints


@test.XFail(reason="Need to refactor Keras model to new API")
def test_KerasBackend_is_trained(clgen_cache_dir, abc_keras_model_config):
  """Test that is_trained changes to True when model is trained."""
  del clgen_cache_dir
  m = models.Model(abc_keras_model_config)
  assert not m.is_trained
  m.Train()
  assert m.is_trained


@test.XFail(reason="Need to refactor Keras model to new API")
def test_KerasBackend_is_trained_new_instance(
  clgen_cache_dir, abc_keras_model_config
):
  """Test that is_trained is True on a new instance of a trained model."""
  del clgen_cache_dir
  m1 = models.Model(abc_keras_model_config)
  m1.Train()
  m2 = models.Model(abc_keras_model_config)
  assert m2.is_trained


# KerasBackend.Train() tests.


@test.XFail(reason="Need to refactor Keras model to new API")
def test_KerasBackend_Train_epoch_checkpoints(
  clgen_cache_dir, abc_keras_model_config
):
  """Test that a trained model generates weight checkpoints."""
  del clgen_cache_dir
  abc_keras_model_config.training.num_epochs = 2
  m = models.Model(abc_keras_model_config)
  m.Train()
  assert len(m.backend.epoch_checkpoints) == 2
  for path in m.backend.epoch_checkpoints:
    assert path.is_file()


@test.XFail(reason="Need to refactor Keras model to new API")
def test_KerasBackend_Train_telemetry(clgen_cache_dir, abc_keras_model_config):
  """Test that model training produced telemetry files."""
  del clgen_cache_dir
  abc_keras_model_config.training.num_epochs = 2
  m = models.Model(abc_keras_model_config)
  assert len(m.TrainingTelemetry()) == 0
  m.Train()
  assert len(m.TrainingTelemetry()) == 2
  for telemetry in m.TrainingTelemetry():
    assert isinstance(telemetry, telemetry_pb2.ModelEpochTelemetry)


@test.XFail(reason="Need to refactor Keras model to new API")
def test_KerasBackend_Train_twice(clgen_cache_dir, abc_keras_model_config):
  """Test that TensorFlow checkpoint does not change after training twice."""
  del clgen_cache_dir
  abc_keras_model_config.training.num_epochs = 1
  m = models.Model(abc_keras_model_config)
  m.Train()
  f1a = checksumdir.dirhash(m.cache.path / "checkpoints")
  f1b = crypto.md5_file(m.cache.path / "META.pbtxt")
  m.Train()
  f2a = checksumdir.dirhash(m.cache.path / "checkpoints")
  f2b = crypto.md5_file(m.cache.path / "META.pbtxt")
  assert f1a == f2a
  assert f1b == f2b


# TODO(cec): Add tests on incrementally trained model predictions and losses.

# KerasBackend.Sample() tests.


@test.XFail(reason="Need to refactor Keras model to new API")
def test_KerasBackend_Sample_implicit_train(
  clgen_cache_dir, abc_keras_model_config
):
  """Test that Sample() implicitly trains the model."""
  del clgen_cache_dir
  m = models.Model(abc_keras_model_config)
  assert not m.is_trained
  m.Sample(MockSampler(), [sample_observers.MaxSampleCountObserver(1)])
  assert m.is_trained


@test.XFail(reason="Need to refactor Keras model to new API")
def test_KerasBackend_Sample_return_value_matches_cached_sample(
  clgen_cache_dir, abc_keras_model_config
):
  """Test that Sample() returns Sample protos."""
  del clgen_cache_dir
  m = models.Model(abc_keras_model_config)
  sample_observer = sample_observers.InMemorySampleSaver()
  m.Sample(
    MockSampler(hash="hash"),
    [sample_observers.MaxSampleCountObserver(1), sample_observer],
  )
  samples = sample_observer.samples
  assert len(samples) == 1
  assert len(list((m.cache.path / "samples" / "hash").iterdir())) == 1
  cached_sample_path = (
    m.cache.path
    / "samples"
    / "hash"
    / list((m.cache.path / "samples" / "hash").iterdir())[0]
  )
  assert cached_sample_path.is_file()
  cached_sample = pbutil.FromFile(cached_sample_path, model_pb2.Sample())
  assert samples[0].text == cached_sample.text
  assert samples[0].sample_time_ms == cached_sample.sample_time_ms
  assert (
    samples[0].sample_start_epoch_ms_utc
    == cached_sample.sample_start_epoch_ms_utc
  )


@test.XFail(reason="Need to refactor Keras model to new API")
def test_KerasBackend_Sample_exact_multiple_of_batch_size(
  clgen_cache_dir, abc_keras_model_config
):
  """Test that min_num_samples are returned when a multiple of batch_size."""
  del clgen_cache_dir
  m = models.Model(abc_keras_model_config)
  sample_observer = sample_observers.InMemorySampleSaver()
  m.Sample(
    MockSampler(batch_size=2),
    [sample_observers.MaxSampleCountObserver(2), sample_observer],
  )
  assert len(sample_observer.samples) == 2
  sample_observer = sample_observers.InMemorySampleSaver()
  m.Sample(
    MockSampler(batch_size=2),
    [sample_observers.MaxSampleCountObserver(4), sample_observer],
  )
  assert len(sample_observer.samples) == 4


@test.XFail(reason="Need to refactor Keras model to new API")
def test_KerasBackend_GetInferenceModel_predict_output_shape(
  clgen_cache_dir, abc_keras_model_config
):
  """Test that predict() on inference model is one-hot encoded."""
  del clgen_cache_dir
  m = models.Model(abc_keras_model_config)
  im, batch_size = m.backend.GetInferenceModel()
  probabilities = im.predict(np.array([[0]]) * batch_size)
  assert (batch_size, 1, m.corpus.vocab_size) == probabilities.shape


# WeightedPick() tests.


def test_WeightedPick_output_range():
  """Test that WeightedPick() returns an integer index into array"""
  a = [1, 2, 3, 4]
  assert 0 <= keras_backend.WeightedPick(np.array(a), 1.0) <= len(a)


# Benchmarks.


@test.XFail(reason="Need to refactor Keras model to new API")
def test_benchmark_KerasBackend_Train_already_trained(
  clgen_cache_dir, abc_keras_model_config, benchmark
):
  """Benchmark the Train() method on an already-trained model."""
  del clgen_cache_dir
  m = models.Model(abc_keras_model_config)
  m.Train()  # "Offline" training from cold.
  benchmark(m.Train)


if __name__ == "__main__":
  test.Main()
