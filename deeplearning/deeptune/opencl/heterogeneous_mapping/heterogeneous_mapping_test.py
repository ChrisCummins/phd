"""Unit tests for //deeplearning/deeptune/opencl/heterogeneous_mapping."""
import pathlib
import tempfile

import pytest
from absl import flags

from deeplearning.deeptune.opencl.heterogeneous_mapping import \
  heterogeneous_mapping
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import \
  models
from labm8 import test

FLAGS = flags.FLAGS


@pytest.fixture(scope='session')
def r() -> heterogeneous_mapping.HeterogeneousMappingExperiment:
  """Test fixture for experimental results."""
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    yield heterogeneous_mapping.HeterogeneousMappingExperiment(pathlib.Path(d))


def test_atomizer_vocab_size(
    r: heterogeneous_mapping.HeterogeneousMappingExperiment):
  """Test that atomizer has expected vocab size."""
  assert r.atomizer.vocab_size == 128


def test_static_mapping_accuracy(
    r: heterogeneous_mapping.HeterogeneousMappingExperiment):
  """Test the accuracy of the static_mapping model."""
  assert r.ResultsDataFrame(
      models.StaticMapping)['Correct?'].mean() == pytest.approx(.5786765)


def test_static_mapping_speedup(
    r: heterogeneous_mapping.HeterogeneousMappingExperiment):
  """Test the speedup of the static_mapping model."""
  # Baseline is Zero-R, and speedup is relative to Zero-R.
  assert r.ResultsDataFrame(
      models.StaticMapping)['Speedup'].mean() == pytest.approx(1)


def test_grewe_accuracy(
    r: heterogeneous_mapping.HeterogeneousMappingExperiment):
  """Test the accuracy of the Grewe et al model."""
  assert r.ResultsDataFrame(
      models.Grewe)['Correct?'].mean() == pytest.approx(.7250)


def test_grewe_speedup(r: heterogeneous_mapping.HeterogeneousMappingExperiment):
  """Test the speedup of the Grewe et al model."""
  assert r.ResultsDataFrame(
      models.Grewe)['Speedup'].mean() == pytest.approx(2.094359)


def test_PrintResultsSummary_smoke_test(
    r: heterogeneous_mapping.HeterogeneousMappingExperiment):
  """Test that PrintResultsSummary() doesn't blow up."""
  df = r.ResultsDataFrame(models.StaticMapping)
  heterogeneous_mapping.HeterogeneousMappingExperiment.PrintResultsSummary(df)


@pytest.mark.slow(reason='Takes several hours to train full model')
def test_deeptune_accuracy(
    r: heterogeneous_mapping.HeterogeneousMappingExperiment):
  """Test the accuracy of the DeepTune model."""
  assert r.ResultsDataFrame(
      models.Deeptune)['Correct?'].mean() == pytest.approx(.819853)


@pytest.mark.slow(reason='Takes several hours to train full model')
def test_deeptune_speedup(
    r: heterogeneous_mapping.HeterogeneousMappingExperiment):
  """Test the speedup of the DeepTune model."""
  assert r.ResultsDataFrame(
      models.Deeptune)['Speedup'].mean() == pytest.approx(2.373917)


@pytest.mark.slow(reason='Takes several hours to train full model')
def test_deeptune_inst2vec_accuracy(
    r: heterogeneous_mapping.HeterogeneousMappingExperiment):
  """Test the accuracy of the DeepTuneInst2Vec model."""
  # TODO(cec): Fill with values after running.
  assert r.ResultsDataFrame(
      models.DeepTuneInst2Vec)['Correct?'].mean() == pytest.approx(0)


@pytest.mark.slow(reason='Takes several hours to train full model')
def test_deeptune_inst2vec_speedup(
    r: heterogeneous_mapping.HeterogeneousMappingExperiment):
  """Test the speedup of the DeepTuneInst2Vec model."""
  # TODO(cec): Fill with values after running.
  assert r.ResultsDataFrame(
      models.DeepTuneInst2Vec)['Speedup'].mean() == pytest.approx(0)


if __name__ == '__main__':
  test.Main()
