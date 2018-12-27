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
from deeplearning.deeptune.opencl.heterogeneous_mapping import \
  models


FLAGS = flags.FLAGS


@pytest.fixture(scope='session')
def r() -> heterogeneous_mapping.ExperimentalResults:
  """Test fixture for experimental results."""
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    yield heterogeneous_mapping.ExperimentalResults(pathlib.Path(d))


@pytest.fixture(scope='function')
def mini_deeptune_model() -> models.DeepTune:
  """Test fixture that returns a small DeepTune model.

  The idea is that this model is quick to train / evaluate.
  """
  return models.DeepTune(num_epochs=1, lstm_layer_size=4, dense_layer_size=4)


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


# TODO(cec): Can we set a flag to run huge tests? Something like
# bazel test //... --test_arg=--phd_run_huge_tests
@pytest.mark.skipif(True, reason='Huge test!')
def test_deeptune_accuracy(r: heterogeneous_mapping.ExperimentalResults):
  """Test the accuracy of the DeepTune model."""
  assert r.deeptune_df['Correct?'].mean() == pytest.approx(
      .819853)


@pytest.mark.skipif(True, reason='Huge test!')
def test_deeptune_speedup(r: heterogeneous_mapping.ExperimentalResults):
  """Test the speedup of the DeepTune model."""
  assert r.deeptune_df['Speedup'].mean() == pytest.approx(
      2.373917)


def test_deeptune_smoke_test(r: heterogeneous_mapping.ExperimentalResults,
                             mini_deeptune_model: models.DeepTune):
  """Test that a small deeptune model can be evaluated."""
  # Test only on the first 30 rows of the full dataset.
  num_rows_to_test_on = 30
  df = r.EvaluateModel(
      mini_deeptune_model, df=r.dataset.df[:num_rows_to_test_on])
  assert len(df) == num_rows_to_test_on
  # Flaky test: it's possible that the model could get *everything* wrong, but
  # this is unlikely.
  assert df['Correct?'].mean() > 0


@pytest.mark.parametrize('table', ('baseline_df', 'grewe_df'))
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
