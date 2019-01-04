"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping."""
import pathlib
import tempfile

import pytest
from absl import flags

from deeplearning.deeptune.opencl.heterogeneous_mapping import \
  heterogeneous_mapping
from deeplearning.deeptune.opencl.heterogeneous_mapping import \
  models
from labm8 import test


FLAGS = flags.FLAGS


@pytest.fixture(scope='session')
def r() -> heterogeneous_mapping.ExperimentalResults:
  """Test fixture for experimental results."""
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    yield heterogeneous_mapping.ExperimentalResults(pathlib.Path(d))


def test_atomizer_vocab_size(r: heterogeneous_mapping.ExperimentalResults):
  """Test that atomizer has expected vocab size."""
  assert r.atomizer.vocab_size == 128


def test_baseline_accuracy(r: heterogeneous_mapping.ExperimentalResults):
  """Test the accuracy of the baseline model."""
  assert r.baseline_df['Correct?'].mean() == pytest.approx(.5786765)


def test_baseline_speedup(r: heterogeneous_mapping.ExperimentalResults):
  """Test the speedup of the baseline model."""
  # Baseline is Zero-R, and speedup is relative to Zero-R.
  assert r.baseline_df['Speedup'].mean() == pytest.approx(1)


def test_grewe_accuracy(r: heterogeneous_mapping.ExperimentalResults):
  """Test the accuracy of the Grewe et al model."""
  assert r.grewe_df['Correct?'].mean() == pytest.approx(.7250)


def test_grewe_speedup(r: heterogeneous_mapping.ExperimentalResults):
  """Test the speedup of the Grewe et al model."""
  assert r.grewe_df['Speedup'].mean() == pytest.approx(2.094359)


@pytest.mark.slow(reason='Takes several hours to train full model')
def test_deeptune_accuracy(r: heterogeneous_mapping.ExperimentalResults):
  """Test the accuracy of the DeepTune model."""
  assert r.deeptune_df['Correct?'].mean() == pytest.approx(
      .819853)


@pytest.mark.slow(reason='Takes several hours to train full model')
def test_deeptune_speedup(r: heterogeneous_mapping.ExperimentalResults):
  """Test the speedup of the DeepTune model."""
  assert r.deeptune_df['Speedup'].mean() == pytest.approx(
      2.373917)


@pytest.mark.slow(reason='Takes 10 minutes')
def test_adversarial_df(r: heterogeneous_mapping.ExperimentalResults):
  """Test that adversarial dataframe can be produced."""
  assert len(r.adversarial_df)


@pytest.mark.parametrize('table', ('baseline_df', 'grewe_df'))
def test_cached_model_evaluation_results_are_equal(
    r: heterogeneous_mapping.ExperimentalResults, table: str):
  """Test that multiple accesses to tables yields equal results."""
  df1 = getattr(r, table)
  df2 = getattr(r, table)
  assert df1['Correct?'].mean() == pytest.approx(df2['Correct?'].mean())


if __name__ == '__main__':
  test.Main()
