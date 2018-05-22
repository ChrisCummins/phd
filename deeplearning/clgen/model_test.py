"""Unit tests for //deeplearning/clgen/model.py."""
import pathlib
import sys

import pytest
from absl import app

from deeplearning.clgen import model
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.tests import testlib as tests
from lib.labm8 import crypto
from lib.labm8 import fs


@pytest.fixture(scope='function')
def abc_corpus_config(abc_corpus):
  """The proto config for a simple Corpus."""
  return corpus_pb2.Corpus(language="opencl", path=abc_corpus,
                           ascii_character_atomizer=True, sequence_length=2,
                           contentfile_separator='\n\n')


@pytest.fixture(scope='function')
def abc_model_config(abc_corpus_config):
  """The proto config for a simple Model."""
  architecture = model_pb2.NetworkArchitecture(
    neuron_type=model_pb2.NetworkArchitecture.LSTM, neurons_per_layer=8,
    num_layers=2)
  training = model_pb2.TrainingOptions(num_epochs=1,
                                       shuffle_corpus_contentfiles_between_epochs=False,
                                       batch_size=5, gradient_clip=5,
                                       initial_learning_rate=0.001,
                                       percent_learning_rate_decay_per_epoch=5,
                                       save_intermediate_checkpoints=False)
  return model_pb2.Model(corpus=abc_corpus_config, architecture=architecture,
                         training=training)


# The Model.hash for a Model instance of abc_model_config.
ABC_MODEL_HASH = 'ad3bd344f2083061c2ad3b800d75c2fa4cc2b1aa'


def test_Model_hash(clgen_cache_dir, abc_model_config):
  """Test that the ID of a known corpus matches expected value."""
  del clgen_cache_dir
  m = model.Model(abc_model_config)
  assert ABC_MODEL_HASH == m.hash


def test_Model_config_hash_different_options(clgen_cache_dir, abc_model_config):
  """Test that model options produce different model hashes."""
  del clgen_cache_dir
  abc_model_config.architecture.neuron_type = model_pb2.NetworkArchitecture.GRU
  m1 = model.Model(abc_model_config)
  abc_model_config.architecture.neuron_type = model_pb2.NetworkArchitecture.RNN
  m2 = model.Model(abc_model_config)
  assert m1.hash != m2.hash


def test_Model_config_hash_different_corpus(clgen_cache_dir, abc_model_config):
  """Test that different corpuses produce different model hashes."""
  del clgen_cache_dir
  abc_model_config.corpus.sequence_length = 5
  m1 = model.Model(abc_model_config)
  abc_model_config.corpus.sequence_length = 10
  m2 = model.Model(abc_model_config)
  assert m1.hash != m2.hash


def test_Model_equality(clgen_cache_dir, abc_model_config):
  """Test that two corpuses with identical options are equivalent."""
  del clgen_cache_dir
  m1 = model.Model(abc_model_config)
  m2 = model.Model(abc_model_config)
  assert m1 == m2


def test_Model_inequality(clgen_cache_dir, abc_model_config):
  """Test that two corpuses with different options are not equivalent."""
  del clgen_cache_dir
  abc_model_config.architecture.num_layers = 1
  m1 = model.Model(abc_model_config)
  abc_model_config.architecture.num_layers = 2
  m2 = model.Model(abc_model_config)
  assert m1 != m2


def test_Model_checkpoint_path_untrained(clgen_cache_dir, abc_model_config):
  """Test that an untrained model has no checkpoint_path."""
  del clgen_cache_dir
  m = model.Model(abc_model_config)
  assert not m.most_recent_checkpoint_path


def test_Model_checkpoint_path_trained(clgen_cache_dir, abc_model_config):
  """Test that a trained model has a TensorFlow checkpoint."""
  del clgen_cache_dir
  m = model.Model(abc_model_config)
  m.Train()
  assert m.most_recent_checkpoint_path
  path = pathlib.Path(m.most_recent_checkpoint_path)
  assert fs.isfile(path / 'checkpoint')
  assert fs.isfile(path / 'META')


def test_Model_train_twice(clgen_cache_dir, abc_model_config):
  """Test that TensorFlow checkpoint does not change after training twice."""
  del clgen_cache_dir
  m = model.Model(abc_model_config)
  m.Train()
  path = pathlib.Path(m.most_recent_checkpoint_path)
  f1a = crypto.md5_file(path / 'checkpoint')
  f1b = crypto.md5_file(path / 'META')
  m.Train()
  f2a = crypto.md5_file(path / 'checkpoint')
  f2b = crypto.md5_file(path / 'META')
  assert f1a == f2a
  assert f1b == f2b


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)

a = tests
