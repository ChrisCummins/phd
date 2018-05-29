"""Unit tests for //deeplearning/clgen/models/data_generators.py."""
import sys

import pytest
from absl import app

from deeplearning.clgen import errors
from deeplearning.clgen.models import data_generators


class CorpusMock(object):

  def __init__(self, corpus_length: int = 100, vocabulary_size: int = 10):
    self.corpus_len = corpus_length
    self.vocabulary_size = vocabulary_size

  def GetTrainingData(self, *args, **kwargs):
    return [1] * self.corpus_len


# DataGeneratorBase tests.

def test_DataGeneratorBase_sequence_length(abc_model_config):
  """Test that sequence length derives from TrainingOptions.sequence_length."""
  opt = abc_model_config.training
  opt.sequence_length = 10
  dg = data_generators.DataGeneratorBase(
      CorpusMock(corpus_length=100), opt)
  assert dg.sequence_length == 10


def test_DataGeneratorBase_sequence_length_too_large(abc_model_config):
  """Test that sequence length derives from TrainingOptions.sequence_length."""
  opt = abc_model_config.training
  opt.sequence_length = 50
  with pytest.raises(errors.UserError) as e_info:
    data_generators.DataGeneratorBase(CorpusMock(corpus_length=10), opt)
  assert ("Requested training.sequence_length (50) is larger than the corpus "
          "(10). Reduce the sequence length to <= 9.") == str(e_info.value)


def test_DataGeneratorBase_batch_size(abc_model_config):
  """Test that batch size is derived from TrainingOptions.batch_size."""
  opt = abc_model_config.training
  opt.batch_size = 10
  dg = data_generators.DataGeneratorBase(CorpusMock(corpus_length=100), opt)
  assert dg.batch_size == 10


def test_DataGeneratorBase_batch_size_too_large(abc_model_config):
  """Test that batch size is reduced when larger than corpus."""
  opt = abc_model_config.training
  opt.batch_size = 50
  opt.sequence_length = 5
  dg = data_generators.DataGeneratorBase(CorpusMock(corpus_length=10), opt)
  assert dg.batch_size == 4


# Vectorize() tests.
def test_Vectorize_empty_input(abc_model_config):
  dg = data_generators.DataGeneratorBase(
      CorpusMock(), abc_model_config.training)
  with pytest.raises(IndexError):
    dg.Vectorize(data_generators.DataBatch([], []))


def test_Vectorize_123(abc_model_config):
  opt = abc_model_config.training
  opt.batch_size = 1
  opt.sequence_length = 3
  dg = data_generators.DataGeneratorBase(CorpusMock(vocabulary_size=3), opt)
  data = dg.Vectorize(data_generators.DataBatch([[0, 1, 2]], [0]))
  # Output is a batch_size * sequence_length * vocabulary size array.
  assert data.X.shape == (1, 3, 3)
  # assert data.X == np.array(
  #     [[[True, False, False], [False, True, False], [False, False, True]]])
  # assert data.y = np.array([True, False, False])


# LazyVectorizingGenerator tests.

# Benchmarks.

@pytest.mark.parametrize('sequence_length', [50, ])
@pytest.mark.parametrize('batch_size', [64, ])
@pytest.mark.parametrize('vocabulary_size', [100, 200])
def test_benchmark_Vectorize(benchmark, abc_model_config,
                             sequence_length, batch_size, vocabulary_size):
  opts = abc_model_config.training
  opts.batch_size = batch_size
  opts.sequence_length = sequence_length
  dg = data_generators.DataGeneratorBase(
      CorpusMock(corpus_length=1000, vocabulary_size=vocabulary_size), opts)
  for _ in range(batch_size):
    x = [[1] * sequence_length] * batch_size
    y = [2] * batch_size
  benchmark(dg.Vectorize, data_generators.DataBatch(x, y))


@pytest.mark.parametrize('sequence_length', [50, ])
@pytest.mark.parametrize('batch_size', [64, ])
@pytest.mark.parametrize('vocabulary_size', [100, 200])
def test_benchmark_LazyVectorizingGenerator(benchmark, abc_model_config,
                                            sequence_length, batch_size,
                                            vocabulary_size):
  opts = abc_model_config.training
  opts.batch_size = batch_size
  opts.sequence_length = sequence_length
  dg = data_generators.LazyVectorizingGenerator(
      CorpusMock(corpus_length=10000, vocabulary_size=vocabulary_size), opts)
  benchmark(dg.__next__)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
