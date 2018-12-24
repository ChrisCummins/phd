"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping."""
import pathlib
import sys
import tempfile
import typing

import pytest
from absl import app
from absl import flags

from deeplearning.deeptune.opencl.heterogeneous_mapping import \
  heterogeneous_mapping


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
  assert r.grewe_df['Correct?'].mean() == pytest.approx(.731618)


def test_grewe_accuracy(r: heterogeneous_mapping.ExperimentalResults):
  """Test the speedup of the Grewe et al model."""
  assert r.grewe_df['Speedup'].mean() == pytest.approx(2.0853115)


def test_deeptune_accuracy(r: heterogeneous_mapping.ExperimentalResults):
  """Test the accuracy of the DeepTune model."""
  assert r.deeptune_df['Correct?'].mean() == pytest.approx(.819853)


def test_deeptune_accuracy(r: heterogeneous_mapping.ExperimentalResults):
  """Test the speedup of the DeepTune model."""
  assert r.deeptune_df['Speedup'].mean() == pytest.approx(2.373917)


@pytest.mark.parametrize('table', ('baseline_df',))
def test_cached_model_evaluation_results_are_equal(
    r: heterogeneous_mapping.ExperimentalResults, table: str):
  """Test that multiple accesses to tables yields equal results."""
  df1 = getattr(r, table)
  df2 = getattr(r, table)
  assert df1['Correct?'].mean() == pytest.approx(df2['Correct?'].mean())


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
