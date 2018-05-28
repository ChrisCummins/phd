"""Pytest fixtures for CLgen unit tests."""
import os
import pathlib
import tarfile
import tempfile

import pytest
from absl import flags

from deeplearning.clgen import dbutil
from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from lib.labm8 import pbutil


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
                           ascii_character_atomizer=True,
                           contentfile_separator='\n\n')


@pytest.fixture(scope='function')
def abc_model_config(abc_corpus_config):
  """The proto config for a simple Model."""
  architecture = model_pb2.NetworkArchitecture(
      neuron_type=model_pb2.NetworkArchitecture.LSTM,
      neurons_per_layer=4,
      num_layers=1)
  optimizer = model_pb2.AdamOptimizer(
      initial_learning_rate_micros=2000,
      learning_rate_decay_per_epoch_micros=5000,
      beta_1_micros=900000,
      beta_2_micros=999000,
      normalized_gradient_clip_micros=5000000)
  training = model_pb2.TrainingOptions(
      num_epochs=1,
      sequence_length=10,
      batch_size=5,
      shuffle_corpus_contentfiles_between_epochs=False,
      adam_optimizer=optimizer)
  return model_pb2.Model(corpus=abc_corpus_config, architecture=architecture,
                         training=training)


@pytest.fixture(scope='function')
def abc_sampler_config():
  """The sampler config for a simple Sampler."""
  maxlen = sampler_pb2.MaxTokenLength(maximum_tokens_in_sample=5)
  sample_stop = [sampler_pb2.SampleTerminationCriterion(maxlen=maxlen)]
  return sampler_pb2.Sampler(start_text='a', batch_size=5, seed=0,
                             termination_criteria=sample_stop)


@pytest.fixture(scope='function')
def empty_db_path() -> str:
  """A text fixture which returns an empty database."""
  with tempfile.TemporaryDirectory(prefix='clgen_') as d:
    db_path = pathlib.Path(d) / 'test.db'
    dbutil.create_db(str(db_path), github=False)
    yield str(db_path)


@pytest.fixture(scope='function')
def abc_db_path(empty_db_path) -> str:
  """A text fixture which returns a database containing three ContentFiles."""
  db = dbutil.connect(empty_db_path)
  c = db.cursor()
  dbutil.sql_insert_dict(c, 'ContentFiles', {'id': 'a', 'contents': 'foo'})
  dbutil.sql_insert_dict(c, 'ContentFiles', {'id': 'b', 'contents': 'bar'})
  dbutil.sql_insert_dict(c, 'ContentFiles', {'id': 'c', 'contents': 'car'})
  c.close()
  db.commit()
  db.close()
  return empty_db_path


@pytest.fixture(scope='function')
def abc_instance_config(clgen_cache_dir, abc_model_config,
                        abc_sampler_config) -> clgen_pb2.Instance:
  """A test fixture that returns an Instance config proto."""
  return clgen_pb2.Instance(working_dir=clgen_cache_dir,
                            model=abc_model_config, sampler=abc_sampler_config)


@pytest.fixture(scope='function')
def abc_instance_file(abc_instance_config) -> str:
  """A test fixture that returns a path to an Instance config file."""
  with tempfile.NamedTemporaryFile() as f:
    pbutil.ToFile(abc_instance_config, pathlib.Path(f.name))
    yield f.name
