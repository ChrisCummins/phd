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
"""Unit tests for //deeplearning/clgen/models/data_generators.py."""
import numpy as np
import pytest

from deeplearning.clgen import errors
from deeplearning.clgen.models import data_generators
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


class CorpusMock(object):

  def __init__(self, corpus_length: int = 100, vocabulary_size: int = 10):
    self.corpus_len = corpus_length
    self.vocab_size = vocabulary_size

  def GetTrainingData(self, *args, **kwargs):
    """Mock to return encoded training data."""
    del args
    del kwargs
    return np.array([1] * self.corpus_len)


# BatchGenerator() tests.
@pytest.mark.skip(reason='TODO(cec):')
def test_BatchGenerator_sequence_length_too_large(abc_model_config):
  """Test that sequence length derives from TrainingOptions.sequence_length."""
  opt = abc_model_config.training
  opt.sequence_length = 50
  with pytest.raises(errors.UserError) as e_info:
    data_generators.BatchGenerator(CorpusMock(corpus_length=10), opt)
  assert ("Requested training.sequence_length (50) is larger than the corpus "
          "(10). Reduce the sequence length to <= 9.") == str(e_info.value)


@pytest.mark.skip(reason='TODO(cec):')
def test_BatchGenerator_batch_size_too_large(abc_model_config):
  """Test that batch size is reduced when larger than corpus."""
  opt = abc_model_config.training
  opt.batch_size = 50
  opt.sequence_length = 5
  with pytest.raises(errors.UserError) as e_info:
    data_generators.BatchGenerator(CorpusMock(corpus_length=10), opt)
  assert ('') == str(e_info.value)


# OneHotEncode() tests.


def test_OneHotEncode_empty_input():
  """Test that OneHotEncode() rejects an empty input."""
  with pytest.raises(IndexError):
    data_generators.OneHotEncode(np.array([]), 3)


def test_OneHotEncode_dtype():
  """Test that OneHotEncode() returns a float array."""
  data = data_generators.OneHotEncode(np.array([0, 1, 2]), 3)
  assert data.dtype == np.float64


def test_OneHotEncode_shape():
  """Test that OneHotEncode() adds a dimension of vocabulary_size."""
  data = data_generators.OneHotEncode(np.array([0, 1, 2]), 4)
  assert data.shape == (3, 4)


def test_OneHotEncode_values():
  """Test that OneHotEncode() returns a correct values for a known input."""
  data = data_generators.OneHotEncode(np.array([0, 1, 2]), 4)
  np.testing.assert_array_equal(
      np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]), data)


# Benchmarks.


@pytest.mark.parametrize('sequence_length', [50, 100, 500, 1000])
@pytest.mark.parametrize('vocabulary_size', [100, 200])
def test_benchmark_OneHotEncode(benchmark, sequence_length, vocabulary_size):
  data = np.zeros(sequence_length, dtype=np.int32)
  benchmark(data_generators.OneHotEncode, data, vocabulary_size)


if __name__ == '__main__':
  test.Main()
