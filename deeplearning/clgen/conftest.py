"""Pytest fixtures for CLgen unit tests."""
import os
import pathlib
import tarfile
import tempfile

import pytest
from absl import flags

from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def clgen_cache_dir() -> str:
  """Creates a temporary directory and sets CLGEN_CACHE to it.

  This fixture has session scope, meaning that the clgen cache directory
  is shared by all unit tests.

  Returns:
    The location of $CLGEN_CACHE.
  """
  with tempfile.TemporaryDirectory(prefix='clgen_cache_') as d:
    os.environ['CLGEN_CACHE'] = d
    yield d


@pytest.fixture(scope='function')
def abc_corpus() -> str:
  """A corpus consisting of three files.

  This fixture has function scope, meaning that a new corpus is created for
  every function which uses this fixture.

  Returns:
    The location of the corpus directory.
  """
  with tempfile.TemporaryDirectory(prefix='clgen_abc_corpus_') as d:
    path = pathlib.Path(d)
    with open(path / 'a', 'w') as f:
      f.write('The cat sat on the mat.')
    with open(path / 'b', 'w') as f:
      f.write('Hello, world!')
    with open(path / 'c', 'w') as f:
      f.write('\nSuch corpus.\nVery wow.')
    yield d


@pytest.fixture(scope='function')
def abc_corpus_archive(abc_corpus) -> str:
  """Creates a .tar.bz2 packed version of the abc_corpus.

  Returns:
    Path to the abc_corpus tarball.
  """
  with tempfile.TemporaryDirectory() as d:
    with tarfile.open(d + '/corpus.tar.bz2', 'w:bz2') as f:
      f.add(abc_corpus + '/a', arcname='corpus/a')
      f.add(abc_corpus + '/b', arcname='corpus/b')
      f.add(abc_corpus + '/c', arcname='corpus/c')
    yield d + '/corpus.tar.bz2'


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
