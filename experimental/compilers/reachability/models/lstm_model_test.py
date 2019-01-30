"""Unit tests for //experimental/compilers/reachability:lstm_model."""
import pathlib

import pandas as pd
import pytest
from absl import flags

from deeplearning.clgen.corpuses import atomizers
from experimental.compilers.reachability import control_flow_graph
from experimental.compilers.reachability.models import lstm_model
from labm8 import test


FLAGS = flags.FLAGS


@pytest.fixture(scope='module')
def atomizer() -> atomizers.AtomizerBase:
  """Test fixture that returns a tiny atomizer."""
  return atomizers.AsciiCharacterAtomizer.FromText(' ABCD')


@pytest.fixture(scope='function')
def df() -> pd.DataFrame:
  """Test fixture that returns a minimal valid dataframe."""
  g = control_flow_graph.ControlFlowGraph(name="A")
  g.add_node(0, name='A')
  g.add_node(1, name='B')
  g.add_edge(0, 1)

  return pd.DataFrame([
    {
      'cfg:block_count': 2,
      'cfg:graph': g.copy(),
      'split:type': 'training',
    },
    {
      'cfg:block_count': 2,
      'cfg:graph': g.copy(),
      'split:type': 'validation',
    },
    {
      'cfg:block_count': 2,
      'cfg:graph': g.copy(),
      'split:type': 'test',
    },
  ])


def test_BuildKerasModel_smoke_test(atomizer: atomizers.AtomizerBase):
  """Test that BuildKerasModel() doesn't blow up."""
  assert lstm_model.BuildKerasModel(
      sequence_length=10, num_classes=3, lstm_size=16, dnn_size=4,
      atomizer=atomizer)


def test_LstmReachabilityModel_TrainAndEvaluate_smoke_test(
    df: pd.DataFrame, tempdir: pathlib.Path):
  """Test that training doesn't blow up."""
  model = lstm_model.LstmReachabilityModel(
      df, tempdir, num_classes=2, lstm_size=8, dnn_size=4)
  model.TrainAndEvaluate(num_epochs=2)


if __name__ == '__main__':
  test.Main()
