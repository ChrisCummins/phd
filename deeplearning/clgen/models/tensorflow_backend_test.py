"""Unit tests for //deeplearning/clgen/models/tensorflow_backend.py."""
import sys

import checksumdir
import numpy as np
import pytest
from absl import app

from deeplearning.clgen.models import models
from deeplearning.clgen.models import tensorflow_backend
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import telemetry_pb2
from lib.labm8 import crypto
from lib.labm8 import pbutil


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


@pytest.mark.skip(reason='TODO(cec): Update checkpoints API.')
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


# TODO(cec): Add tests on incrementally trained model predictions and losses.


# TensorFlowBackend.Sample() tests.

@pytest.mark.skip(reason='TODO(cec): Implement is_trained.')
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


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
