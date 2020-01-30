# Copyright (c) 2016-2020 Chris Cummins.
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
"""Unit tests for //deeplearning/clgen/models/models.py."""
import pathlib

from deeplearning.clgen import errors
from deeplearning.clgen.models import models
from deeplearning.clgen.proto import internal_pb2
from deeplearning.clgen.proto import model_pb2
from labm8.py import app
from labm8.py import pbutil
from labm8.py import test

FLAGS = app.FLAGS

pytest_plugins = ["deeplearning.clgen.tests.fixtures"]

# The Model.hash for an instance of abc_model_config.
ABC_MODEL_HASH = "c1b6ccf7f4136aaafdf9ac61392ddd9061315a83"


def test_Model_config_type_error():
  """Test that a TypeError is raised if config is not a Model proto."""
  with test.Raises(TypeError) as e_info:
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


def test_Model_config_hash_different_num_epochs(
  clgen_cache_dir, abc_model_config
):
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
  abc_model_config.corpus.contentfile_separator = "\n\n"
  m1 = models.Model(abc_model_config)
  abc_model_config.corpus.contentfile_separator = "abc"
  m2 = models.Model(abc_model_config)
  assert m1.hash != m2.hash


def test_Model_config_sequence_length_not_set(
  clgen_cache_dir, abc_model_config
):
  """Test that an error is raised if sequence_length is < 1."""
  del clgen_cache_dir
  abc_model_config.training.sequence_length = -1
  with test.Raises(errors.UserError):
    models.Model(abc_model_config)


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
  assert (m.cache.path / "checkpoints").is_dir()
  assert (m.cache.path / "samples").is_dir()
  # There should be nothing in these directories yet.
  assert not list((m.cache.path / "checkpoints").iterdir())
  assert not list((m.cache.path / "samples").iterdir())


def test_Model_metafile(clgen_cache_dir, abc_model_config):
  """A newly instantiated model's cache has a metafile."""
  del clgen_cache_dir
  m = models.Model(abc_model_config)
  assert (m.cache.path / "META.pbtxt").is_file()
  assert pbutil.ProtoIsReadable(
    m.cache.path / "META.pbtxt", internal_pb2.ModelMeta()
  )


def test_Model_corpus_symlink(clgen_cache_dir, abc_model_config):
  """Test path of symlink to corpus files."""
  del clgen_cache_dir
  m = models.Model(abc_model_config)
  assert (m.cache.path / "corpus").is_symlink()
  path = str((m.cache.path / "corpus").resolve())
  # We can't do a literal comparison because of bazel sandboxing.
  assert path.endswith(
    str(pathlib.Path(m.corpus.encoded.url[len("sqlite:///") :]).parent)
  )


def test_Model_atomizer_symlink(clgen_cache_dir, abc_model_config):
  """Test path of symlink to atomizer."""
  del clgen_cache_dir
  m = models.Model(abc_model_config)
  assert (m.cache.path / "atomizer").is_symlink()
  path = str((m.cache.path / "atomizer").resolve())
  # We can't do a literal comparison because of bazel sandboxing.
  assert path.endswith(str(m.corpus.atomizer_path))


# TODO(cec): Add tests on ModelMeta contents.

# TODO(cec): Add tests on log files and stderr logging.

# TODO(cec): Add test where batch_size is larger than corpus.

# Benchmarks.


def test_benchmark_Model_instantiation(
  clgen_cache_dir, abc_model_config, benchmark
):
  """Benchmark model instantiation.

  We can expect the first iteration of this benchmark to take a little more
  time than subsequent iterations since it must create the cache directories.
  """
  del clgen_cache_dir
  benchmark(models.Model, abc_model_config)


if __name__ == "__main__":
  test.Main()
