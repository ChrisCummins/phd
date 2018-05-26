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


# The Model.hash for a Model instance of abc_model_config.
ABC_MODEL_HASH = '98dadcd7890565e65be97ac212a141a744e8b016'


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


def test_Model_metafile(clgen_cache_dir, abc_model_config):
  """Test that a newly instantiated model has a metafile."""
  del clgen_cache_dir
  m = models.Model(abc_model_config)
  assert (m.cache.path / 'META.pbtxt').is_file()
  assert pbutil.ProtoIsReadable(m.cache.path / 'META.pbtxt',
                                internal_pb2.ModelMeta())


# TODO(cec): Add tests on ModelMeta contents.


def test_Model_epoch_checkpoints_untrained(clgen_cache_dir, abc_model_config):
  """Test that an untrained model has no checkpoint files."""
  del clgen_cache_dir
  m = models.Model(abc_model_config)
  assert not m.epoch_checkpoints


def test_Model_checkpoint_path_trained(clgen_cache_dir, abc_model_config):
  """Test that a trained model has a TensorFlow checkpoint."""
  del clgen_cache_dir
  abc_model_config.training.num_epochs = 2
  m = models.Model(abc_model_config)
  m.Train()
  assert len(m.epoch_checkpoints) == 2
  for path in m.epoch_checkpoints:
    assert path.is_file()


def test_Model_train_twice(clgen_cache_dir, abc_model_config):
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


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
