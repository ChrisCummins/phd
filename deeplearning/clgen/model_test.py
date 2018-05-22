"""Unit tests for //deeplearning/clgen/model.py."""
import sys

import pytest
from absl import app

from deeplearning.clgen import model
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.tests import testlib as tests


@pytest.fixture(scope='function')
def abc_corpus_config(abc_corpus):
  """The proto config for a simple Corpus."""
  return corpus_pb2.Corpus(language="opencl", path=abc_corpus,
                           ascii_character_atomizer=True)


@pytest.fixture(scope='function')
def abc_model_config(abc_corpus_config):
  """The proto config for a simple Model."""
  architecture = model_pb2.NetworkArchitecture(
    neuron_type=model_pb2.NetworkArchitecture.LSTM, neurons_per_layer=8,
    num_layers=2)
  training = model_pb2.TrainingOptions(num_epochs=1,
                                       shuffle_corpus_contentfiles_between_epochs=False,
                                       batch_size=128, gradient_clip=5,
                                       initial_learning_rate=0.001,
                                       percent_learning_rate_decay_per_epoch=5,
                                       save_intermediate_checkpoints=False)
  return model_pb2.Model(corpus=abc_corpus_config, architecture=architecture,
                         training=training)


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


a = tests


# def test_model_checkpoint_path_untrained(clgen_cache_dir):
#   del clgen_cache_dir
#   m = get_test_model()
#   m.cache.clear()  # untrain
#   assert m.checkpoint_path == None
#
#
# def test_model_eq(clgen_cache_dir):
#   del clgen_cache_dir
#   m1 = model.Model.from_json(
#     {"corpus": {"language": "opencl", "path": tests.archive("tiny",
# "corpus")},
#      "train_opts": {"intermediate_checkpoints": False}})
#   m2 = model.Model.from_json(
#     {"corpus": {"language": "opencl", "path": tests.archive("tiny",
# "corpus")},
#      "train_opts": {"intermediate_checkpoints": False}})
#   m3 = model.Model.from_json(
#     {"corpus": {"language": "opencl", "path": tests.archive("tiny",
# "corpus")},
#      "train_opts": {"intermediate_checkpoints": True}})
#
#   assert m1 == m2
#   assert m2 != m3
#   assert m1 != False
#   assert m1 != 'abcdef'
#
#
# def test_json_equivalency(clgen_cache_dir):
#   del clgen_cache_dir
#   m1 = model.Model.from_json(
#     {"corpus": {"language": "opencl", "path": tests.archive("tiny",
# "corpus")},
#      "train_opts": {"intermediate_checkpoints": True}})
#   m2 = model.Model.from_json(m1.to_json())
#   assert m1 == m2


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
