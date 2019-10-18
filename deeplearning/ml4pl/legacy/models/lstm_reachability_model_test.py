"""Unit tests for //deeplearning/ml4pl:lstm_model."""
import pandas as pd
import pathlib
import pytest
from deeplearning.ml4pl.models import lstm_reachability_model

from deeplearning.clgen.corpuses import atomizers
from deeplearning.ml4pl.graphs.unlabelled.cfg import control_flow_graph
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


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
  assert lstm_reachability_model.BuildKerasModel(
      sequence_length=10,
      num_classes=3,
      lstm_size=16,
      dnn_size=4,
      atomizer=atomizer)


def test_LstmReachabilityModel_TrainAndEvaluate_smoke_test(
    df: pd.DataFrame, tempdir: pathlib.Path):
  """Test that training doesn't blow up."""
  model = lstm_reachability_model.LstmReachabilityModel(
      df, tempdir, num_classes=2, lstm_size=8, dnn_size=4)
  model.TrainAndEvaluate(num_epochs=2)


def test_ZeroRReachabilityModel_TrainAndEvaluate_smoke_test(
    df: pd.DataFrame, tempdir: pathlib.Path):
  """Test that training doesn't blow up."""
  model = lstm_reachability_model.ZeroRReachabilityModel(
      df, tempdir, num_classes=2)
  acc, solved = model.TrainAndEvaluate(num_epochs=None)

  # In our fixture df, training == test graph, so ZeroR learns on the graph that
  # it tests on. The graph has exactly two nodes with a single edge between
  # them, so max ZeroR accuracy is 50%.
  assert acc == .5
  assert solved == 0


if __name__ == '__main__':
  test.Main()
