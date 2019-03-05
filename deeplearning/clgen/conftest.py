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
"""Pytest fixtures for CLgen unit tests."""
import os
import pathlib
import tarfile
import tempfile

import pytest
from absl import flags

from deeplearning.clgen.proto import clgen_pb2
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.proto import model_pb2
from deeplearning.clgen.proto import sampler_pb2
from labm8 import pbutil

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
  return corpus_pb2.Corpus(
      local_directory=abc_corpus,
      ascii_character_atomizer=True,
      contentfile_separator='\n\n')


@pytest.fixture(scope='function')
def abc_model_config(abc_corpus_config):
  """The proto config for a simple Model."""
  architecture = model_pb2.NetworkArchitecture(
      backend=model_pb2.NetworkArchitecture.TENSORFLOW,
      embedding_size=2,
      neuron_type=model_pb2.NetworkArchitecture.LSTM,
      neurons_per_layer=4,
      num_layers=1,
      post_layer_dropout_micros=2000)
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
  return model_pb2.Model(
      corpus=abc_corpus_config, architecture=architecture, training=training)


@pytest.fixture(scope='function')
def abc_sampler_config():
  """The sampler config for a simple Sampler."""
  maxlen = sampler_pb2.MaxTokenLength(maximum_tokens_in_sample=5)
  sample_stop = [sampler_pb2.SampleTerminationCriterion(maxlen=maxlen)]
  return sampler_pb2.Sampler(
      start_text='a',
      batch_size=5,
      sequence_length=10,
      termination_criteria=sample_stop,
      temperature_micros=1000000)


@pytest.fixture(scope='function')
def abc_instance_config(clgen_cache_dir, abc_model_config,
                        abc_sampler_config) -> clgen_pb2.Instance:
  """A test fixture that returns an Instance config proto."""
  return clgen_pb2.Instance(
      working_dir=clgen_cache_dir,
      model=abc_model_config,
      sampler=abc_sampler_config)


@pytest.fixture(scope='function')
def abc_instance_file(abc_instance_config) -> str:
  """A test fixture that returns a path to an Instance config file."""
  with tempfile.NamedTemporaryFile() as f:
    pbutil.ToFile(abc_instance_config, pathlib.Path(f.name))
    yield f.name
