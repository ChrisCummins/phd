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
"""Unit tests for //deeplearning/clgen/models/tensorflow_backend.py."""

import checksumdir
import numpy as np
import pytest
from absl import flags

from deeplearning.clgen.models import models
from deeplearning.clgen.models import tensorflow_backend
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import telemetry_pb2
from labm8 import crypto
from labm8 import pbutil
from labm8 import test


FLAGS = flags.FLAGS


class MockSampler(object):
  """Mock class for a Sampler."""

  # The default value for start_text has been chosen to only use characters and
  # words from the abc_corpus, so that it may be encoded using the vocabulary
  # of that corpus.
  def __init__(self, start_text: str = 'H', hash: str = 'hash'):
    self.start_text = start_text
    self.encoded_start_text = np.array([1, 2, 3])
    self.tokenized_start_text = ['a', 'b', 'c']
    self.temperature = 1.0
    self.hash = hash

  @staticmethod
  def Specialize(atomizer):
    """Atomizer.Specialize() mock."""
    pass

  @staticmethod
  def SampleIsComplete(sample_in_progress):
    """Crude 'maxlen' mock."""
    return len(sample_in_progress) >= 10


@pytest.fixture(scope='function')
def abc_tensorflow_model_config(abc_model_config: model_pb2.Model):
  """A test fixture for a simple model with a TensorFlow backend."""
  abc_model_config.architecture.backend = model_pb2.NetworkArchitecture.TENSORFLOW
  return abc_model_config


# TensorFlowBackend.Train() tests.

def test_TensorFlowBackend_Train_telemetry(
    clgen_cache_dir, abc_tensorflow_model_config):
  """Test that model training produced telemetry files."""
  del clgen_cache_dir
  abc_tensorflow_model_config.training.num_epochs = 2
  m = models.Model(abc_tensorflow_model_config)
  assert len(m.TrainingTelemetry()) == 0
  m.Train()
  assert len(m.TrainingTelemetry()) == 2
  for telemetry in m.TrainingTelemetry():
    assert isinstance(telemetry, telemetry_pb2.ModelEpochTelemetry)


def test_TensorFlowBackend_Train_twice(
    clgen_cache_dir, abc_tensorflow_model_config):
  """Test that TensorFlow checkpoint does not change after training twice."""
  del clgen_cache_dir
  abc_tensorflow_model_config.training.num_epochs = 1
  m = models.Model(abc_tensorflow_model_config)
  m.Train()
  f1a = checksumdir.dirhash(m.cache.path / 'checkpoints')
  f1b = crypto.md5_file(m.cache.path / 'META.pbtxt')
  m.Train()
  f2a = checksumdir.dirhash(m.cache.path / 'checkpoints')
  f2b = crypto.md5_file(m.cache.path / 'META.pbtxt')
  assert f1a == f2a
  assert f1b == f2b


def test_TensorFlowBackend_Train_epoch_checkpoints(
    clgen_cache_dir, abc_tensorflow_model_config):
  """Test that epoch_checkpoints returns a <int, str> dict."""
  del clgen_cache_dir
  abc_tensorflow_model_config.training.num_epochs = 2
  m = models.Model(abc_tensorflow_model_config)
  assert not m.backend.epoch_checkpoints
  m.Train()
  epoch_checkpoints = m.backend.epoch_checkpoints
  assert 2 == len(epoch_checkpoints)
  assert 1 in epoch_checkpoints
  assert 2 in epoch_checkpoints


def test_TensorFlowBackend_Train_missing_intermediate_checkpoints(
    clgen_cache_dir, abc_tensorflow_model_config):
  """Test that a missing intermediate checkpoint does not affect training."""
  del clgen_cache_dir
  abc_tensorflow_model_config.training.num_epochs = 2
  m = models.Model(abc_tensorflow_model_config)
  m.Train()
  assert 2 == len(m.backend.epoch_checkpoints)

  checkpoints_dir = (m.cache.path / 'checkpoints')
  for path in checkpoints_dir.iterdir():
    # Remove all files which are not either the checkpoints list, or the most
    # recent checkpoint.
    if (not path.name == 'checkpoint' and not
    path.name.startswith('checkpoint-2')):
      path.unlink()
  f1a = checksumdir.dirhash(checkpoints_dir)

  assert 1 == len(m.backend.epoch_checkpoints)
  assert 2 in m.backend.epoch_checkpoints

  # Run Train() again to check that nothing is changed.
  m.Train()
  assert 1 == len(m.backend.epoch_checkpoints)
  assert 2 in m.backend.epoch_checkpoints
  f1b = checksumdir.dirhash(checkpoints_dir)
  assert f1a == f1b


def test_TensorFlowBackend_Train_is_trained(
    clgen_cache_dir, abc_tensorflow_model_config):
  """Test that is_trained is initially false until trained."""
  del clgen_cache_dir
  m = models.Model(abc_tensorflow_model_config)
  assert not m.is_trained
  m.Train()
  assert m.is_trained


# TODO(cec): Add test for InferenceManifest() contents of a simple model.


# TODO(cec): Add tests on incrementally trained model predictions and losses.


# TensorFlowBackend.Sample() tests.

def test_TensorFlowBackend_Sample_implicit_train(clgen_cache_dir,
                                                 abc_tensorflow_model_config):
  """Test that Sample() implicitly trains the model."""
  del clgen_cache_dir
  m = models.Model(abc_tensorflow_model_config)
  assert not m.is_trained
  m.Sample(MockSampler(), 1)
  assert m.is_trained


def test_TensorFlowBackend_Sample_return_value_matches_cached_sample(
    clgen_cache_dir,
    abc_tensorflow_model_config):
  """Test that Sample() returns Sample protos."""
  del clgen_cache_dir
  abc_tensorflow_model_config.training.batch_size = 1
  m = models.Model(abc_tensorflow_model_config)
  samples = m.Sample(MockSampler(hash='hash'), 1)
  assert len(samples) == 1
  assert len(list((m.cache.path / 'samples' / 'hash').iterdir())) == 1
  cached_sample_path = (m.cache.path / 'samples' / 'hash' /
                        list((m.cache.path / 'samples' / 'hash').iterdir())[0])
  assert cached_sample_path.is_file()
  cached_sample = pbutil.FromFile(cached_sample_path, model_pb2.Sample())
  assert samples[0].text == cached_sample.text
  assert samples[0].sample_time_ms == cached_sample.sample_time_ms
  assert samples[
           0].sample_start_epoch_ms_utc == cached_sample.sample_start_epoch_ms_utc


def test_TensorFlowBackend_Sample_exact_multiple_of_batch_size(
    clgen_cache_dir,
    abc_tensorflow_model_config):
  """Test that min_num_samples are returned when a multiple of batch_size."""
  del clgen_cache_dir
  abc_tensorflow_model_config.training.batch_size = 2
  m = models.Model(abc_tensorflow_model_config)
  assert len(m.Sample(MockSampler(), 2)) == 2
  assert len(m.Sample(MockSampler(), 4)) == 4


def test_TensorFlowBackend_Sample_inexact_multiple_of_batch_size(
    clgen_cache_dir,
    abc_tensorflow_model_config):
  """Test that min_num_samples are returned when a multiple of batch_size."""
  del clgen_cache_dir
  abc_tensorflow_model_config.training.batch_size = 3
  m = models.Model(abc_tensorflow_model_config)
  # 3 = 1 * sizeof(batch).
  assert len(m.Sample(MockSampler(), 2)) == 3
  # 6 = 2 * sizeof(batch).
  assert len(m.Sample(MockSampler(), 4)) == 6


# WeightedPick() tests.

@pytest.mark.skip(reason='TODO(cec): Update for new WeightedPick().')
def test_WeightedPick_output_range():
  """Test that WeightedPick() returns an integer index into array"""
  a = [1, 2, 3, 4]
  assert 0 <= tensorflow_backend.WeightedPick(np.array(a), 1.0) <= len(a)


# Benchmarks.

def test_benchmark_TensorFlowModel_Train_already_trained(
    clgen_cache_dir, abc_tensorflow_model_config, benchmark):
  """Benchmark the Train() method on an already-trained model."""
  del clgen_cache_dir
  m = models.Model(abc_tensorflow_model_config)
  m.Train()  # "Offline" training from cold.
  benchmark(m.Train)


if __name__ == '__main__':
  test.Main()
