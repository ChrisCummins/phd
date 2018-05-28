"""Unit tests for //deeplearning/clgen/model.py."""
import sys

import checksumdir
import pytest
from absl import app

from deeplearning.clgen.models import models
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.proto import model_pb2
from lib.labm8 import crypto
from lib.labm8 import pbutil


class MockSampler(object):
  """Mock class for a Sampler."""

  # The default value for start_text has been chosen to only use characters and
  # words from the abc_corpus, so that it may be encoded using the vocabulary
  # of that corpus.
  def __init__(self, start_text: str = 'H', hash: str = 'hash',
               batch_size: int = 1):
    self.start_text = start_text
    self.hash = hash
    self.batch_size = batch_size
    self.has_symmetrical_tokens = False

  @staticmethod
  def SampleIsComplete(sample_in_progress):
    return len(sample_in_progress) >= 10


# The Model.hash for an instance of abc_model_config.
ABC_MODEL_HASH = 'bad4436bf51066ba87b2fcc3505a15030033c9d3'


def test_Model_config_type_error():
  """Test that a TypeError is raised if config is not a Model proto."""
  with pytest.raises(TypeError) as e_info:
    models.Model(1)
  assert "Config must be a Model proto. Received: 'int'" == str(e_info.value)


def test_Model_hash(clgen_cache_dir, abc_model_config):
  """Test that the ID of a known corpus matches expected value."""
  del clgen_cache_dir
  m = models.Model(abc_model_config)
  assert ABC_MODEL_HASH == m.hash


def test_Model_config_hash_different_options(clgen_cache_dir, abc_model_config):
  """Test that model options produce different model hashes."""
  del clgen_cache_dir
  abc_model_config.architecture.neuron_type = model_pb2.NetworkArchitecture.GRU
  m1 = models.Model(abc_model_config)
  abc_model_config.architecture.neuron_type = model_pb2.NetworkArchitecture.RNN
  m2 = models.Model(abc_model_config)
  assert m1.hash != m2.hash


def test_Model_config_hash_different_num_epochs(clgen_cache_dir,
                                                abc_model_config):
  """Test that different num_eopchs doesn't affect model hash."""
  del clgen_cache_dir
  abc_model_config.training.num_epochs = 10
  m1 = models.Model(abc_model_config)
  abc_model_config.training.num_epochs = 20
  m2 = models.Model(abc_model_config)
  assert m1.hash == m2.hash


def test_Model_config_hash_different_corpus(clgen_cache_dir, abc_model_config):
  """Test that different corpuses produce different model hashes."""
  del clgen_cache_dir
  abc_model_config.corpus.sequence_length = 5
  m1 = models.Model(abc_model_config)
  abc_model_config.corpus.sequence_length = 10
  m2 = models.Model(abc_model_config)
  assert m1.hash != m2.hash


def test_Model_equality(clgen_cache_dir, abc_model_config):
  """Test that two corpuses with identical options are equivalent."""
  del clgen_cache_dir
  m1 = models.Model(abc_model_config)
  m2 = models.Model(abc_model_config)
  assert m1 == m2


def test_Model_inequality(clgen_cache_dir, abc_model_config):
  """Test that two corpuses with different options are not equivalent."""
  del clgen_cache_dir
  abc_model_config.architecture.num_layers = 1
  m1 = models.Model(abc_model_config)
  abc_model_config.architecture.num_layers = 2
  m2 = models.Model(abc_model_config)
  assert m1 != m2


def test_Model_directories(clgen_cache_dir, abc_model_config):
  """A newly instantiated model's cache has checkpoint and sample dirs."""
  del clgen_cache_dir
  m = models.Model(abc_model_config)
  assert (m.cache.path / 'checkpoints').is_dir()
  assert (m.cache.path / 'samples').is_dir()
  # There should be nothing in these directories yet.
  assert not list((m.cache.path / 'checkpoints').iterdir())
  assert not list((m.cache.path / 'samples').iterdir())


def test_Model_metafile(clgen_cache_dir, abc_model_config):
  """A newly instantiated model's cache has a metafile."""
  del clgen_cache_dir
  m = models.Model(abc_model_config)
  assert (m.cache.path / 'META.pbtxt').is_file()
  assert pbutil.ProtoIsReadable(m.cache.path / 'META.pbtxt',
                                internal_pb2.ModelMeta())


# TODO(cec): Add tests on ModelMeta contents.

# TODO(cec): Add tests on log files and stderr logging.


def test_Model_epoch_checkpoints_untrained(clgen_cache_dir, abc_model_config):
  """Test that an untrained model has no checkpoint files."""
  del clgen_cache_dir
  m = models.Model(abc_model_config)
  assert not m.epoch_checkpoints


# Model.Train() tests.


def test_Model_is_trained(clgen_cache_dir, abc_model_config):
  """Test that is_trained changes to True when model is trained."""
  del clgen_cache_dir
  m = models.Model(abc_model_config)
  assert not m.is_trained
  m.Train()
  assert m.is_trained


def test_Model_is_trained_new_instance(clgen_cache_dir, abc_model_config):
  """Test that is_trained is True on a new instance of a trained model."""
  del clgen_cache_dir
  m1 = models.Model(abc_model_config)
  m1.Train()
  m2 = models.Model(abc_model_config)
  assert m2.is_trained


def test_Model_Train_epoch_checkpoints(clgen_cache_dir, abc_model_config):
  """Test that a trained model has a TensorFlow checkpoint."""
  del clgen_cache_dir
  abc_model_config.training.num_epochs = 2
  m = models.Model(abc_model_config)
  m.Train()
  assert len(m.epoch_checkpoints) == 2
  for path in m.epoch_checkpoints:
    assert path.is_file()


def test_Model_Train_twice(clgen_cache_dir, abc_model_config):
  """Test that TensorFlow checkpoint does not change after training twice."""
  del clgen_cache_dir
  abc_model_config.training.num_epochs = 1
  m = models.Model(abc_model_config)
  m.Train()
  f1a = checksumdir.dirhash(m.cache.path / 'checkpoints')
  f1b = crypto.md5_file(m.cache.path / 'META.pbtxt')
  m.Train()
  f2a = checksumdir.dirhash(m.cache.path / 'checkpoints')
  f2b = crypto.md5_file(m.cache.path / 'META.pbtxt')
  assert f1a == f2a
  assert f1b == f2b


# TODO(cec): Add tests on incrementally trained model predictions and losses.

# TODO(cec): Add test where batch_size is larger than corpus.

# Model.Sample() tests.


def test_Model_Sample_invalid_start_text(clgen_cache_dir, abc_model_config):
  """Test that InvalidStartText error raised if start text cannot be encoded."""
  del clgen_cache_dir
  m = models.Model(abc_model_config)
  with pytest.raises(errors.InvalidStartText):
    # This start_text chosen to include characters not in the abc_corpus.
    m.Sample(MockSampler(start_text='!@#1234'), 1)


def test_Model_Sample_implicit_train(clgen_cache_dir, abc_model_config):
  """Test that Sample() implicitly trains the model."""
  del clgen_cache_dir
  m = models.Model(abc_model_config)
  assert not m.is_trained
  m.Sample(MockSampler(), 1)
  assert m.is_trained


def test_Model_Sample_return_value_matches_cached_sample(clgen_cache_dir,
                                                         abc_model_config):
  """Test that Sample() returns Sample protos."""
  del clgen_cache_dir
  m = models.Model(abc_model_config)
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


def test_Model_Sample_exact_multiple_of_batch_size(clgen_cache_dir,
                                                   abc_model_config):
  """Test that min_num_samples are returned when a multiple of batch_size."""
  del clgen_cache_dir
  m = models.Model(abc_model_config)
  assert len(m.Sample(MockSampler(batch_size=2), 2)) == 2
  assert len(m.Sample(MockSampler(batch_size=2), 4)) == 4


# Benchmarks.

def test_benchmark_Model_instantiation(clgen_cache_dir, abc_model_config,
                                       benchmark):
  """Benchmark model instantiation.

  We can expect the first iteration of this benchmark to take a little more
  time than subsequent iterations since it must create the cache directories.
  """
  del clgen_cache_dir
  benchmark(models.Model, abc_model_config)


def test_benchmark_Model_Train_already_trained(clgen_cache_dir,
                                               abc_model_config, benchmark):
  """Benchmark the Train() method on an already-trained model."""
  del clgen_cache_dir
  m = models.Model(abc_model_config)
  m.Train()  # "Offline" training from cold.
  benchmark(m.Train)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
