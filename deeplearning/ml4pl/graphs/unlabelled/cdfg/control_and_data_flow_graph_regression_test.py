"""Test graph builder on bytecodes that were found to exposure bugs."""
import pytest
from labm8 import app
from labm8 import bazelutil
from labm8 import fs
from labm8 import test

from deeplearning.ml4pl.graphs.unlabelled.cdfg import \
  control_and_data_flow_graph as cdfg

FLAGS = app.FLAGS

MODULE_UNDER_TEST = 'deeplearning.ml4pl.graphs.unlabelled.cdfg'

REGRESSION_TESTS = bazelutil.DataPath(
    'phd/deeplearning/ml4pl/graphs/unlabelled/cdfg/regression_tests')


@pytest.fixture(scope='function')
def builder() -> cdfg.ControlAndDataFlowGraphBuilder:
  """Test fixture that returns the graph builder for regression tests."""
  return cdfg.ControlAndDataFlowGraphBuilder()


def test_105975(builder: cdfg.ControlAndDataFlowGraphBuilder):
  """CFG has BBs without predecessors that need to be removed."""
  builder.Build(fs.Read(REGRESSION_TESTS / '105975.ll'))


def _IsTimeout(err, *args):
  del args
  return isinstance(err[0], TimeoutError)


# This is a large graph which may timeout on a loaded / slow system.
@pytest.mark.flaky(max_runs=3, rerun_filter=_IsTimeout)
def test_115532(builder: cdfg.ControlAndDataFlowGraphBuilder):
  """Number of callsites does not correlate with callgraph."""
  builder.Build(fs.Read(REGRESSION_TESTS / '115532.ll'))


@pytest.mark.xfail(reason='Timeout')
def test_4180(builder: cdfg.ControlAndDataFlowGraphBuilder):
  """Graph takes more than 120 seconds to construct."""
  builder.Build(fs.Read(REGRESSION_TESTS / '4180.ll'))


# TODO(cec): Add support for functions without exit blocks. They will have
# no call return edges.
@pytest.mark.xfail(reason='Cannot currently handle no exit blocks')
def test_560(builder: cdfg.ControlAndDataFlowGraphBuilder):
  """Graph has no exit blocks."""
  builder.Build(fs.Read(REGRESSION_TESTS / '560.ll'))


@pytest.mark.xfail(reason='opt exception')
def test_400531(builder: cdfg.ControlAndDataFlowGraphBuilder):
  """Graph has no exit blocks."""
  builder.Build(fs.Read(REGRESSION_TESTS / '400531.ll'))


if __name__ == '__main__':
  test.Main()
