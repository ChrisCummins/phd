"""Test graph builder on bytecodes that were found to exposure bugs."""
from deeplearning.ml4pl.graphs.unlabelled.llvm2graph import graph_builder
from labm8.py import bazelutil
from labm8.py import fs
from labm8.py import test

FLAGS = test.FLAGS

MODULE_UNDER_TEST = "deeplearning.ml4pl.graphs.unlabelled.llvm2graph"

REGRESSION_TESTS = bazelutil.DataPath(
  "phd/deeplearning/ml4pl/testing/data/bytecode_regression_tests"
)


@test.Fixture(scope="function")
def builder() -> graph_builder.ProGraMLGraphBuilder:
  """Test fixture that returns the graph builder for regression tests."""
  return graph_builder.ProGraMLGraphBuilder()


@test.XFail(reason="TODO(github.com/ChrisCummins/ProGraML/issues/2)")
def test_105975(builder: graph_builder.ProGraMLGraphBuilder):
  """CFG has BBs without predecessors that need to be removed."""
  builder.Build(fs.Read(REGRESSION_TESTS / "105975.ll"))


@test.XFail(reason="TODO(github.com/ChrisCummins/ProGraML/issues/2)")
@test.Flaky(
  max_runs=5,
  expected_exception=TimeoutError,
  reason="This is a large graph which may timeout on a loaded system.",
)
def test_115532(builder: graph_builder.ProGraMLGraphBuilder):
  """Number of callsites does not correlate with callgraph."""
  builder.Build(fs.Read(REGRESSION_TESTS / "115532.ll"))


@test.XFail(reason="TODO(github.com/ChrisCummins/ProGraML/issues/2)")
def test_4180(builder: graph_builder.ProGraMLGraphBuilder):
  """Graph takes more than 120 seconds to construct."""
  builder.Build(fs.Read(REGRESSION_TESTS / "4180.ll"))


@test.XFail(reason="TODO(github.com/ChrisCummins/ProGraML/issues/2)")
# TODO(github.com/ChrisCummins/ProGraML/issues/11): Add support for functions
# without exit blocks. They will have no call return edges.
# @test.XFail(reason="Cannot currently handle no exit blocks")
def test_560(builder: graph_builder.ProGraMLGraphBuilder):
  """Graph has no exit blocks."""
  builder.Build(fs.Read(REGRESSION_TESTS / "560.ll"))


@test.XFail(reason="TODO(github.com/ChrisCummins/ProGraML/issues/2)")
# @test.XFail(reason="opt exception")
def test_400531(builder: graph_builder.ProGraMLGraphBuilder):
  """Graph has no exit blocks."""
  builder.Build(fs.Read(REGRESSION_TESTS / "400531.ll"))


if __name__ == "__main__":
  test.Main()
