"""Test the annotate binary."""
from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow import annotate
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import bazelutil
from labm8.py import test

FLAGS = test.FLAGS

# This tests the annotate library as a binary, no test coverage.
MODULE_UNDER_TEST = None

###############################################################################
# Fixtures.
###############################################################################


@test.Fixture(
  scope="session",
  params=list(random_programl_generator.EnumerateProtoTestSet()),
)
def real_proto(request) -> programl_pb2.ProgramGraph:
  """A test fixture which enumerates one of 100 "real" protos."""
  return request.param


@test.Fixture(scope="session")
def one_proto() -> programl_pb2.ProgramGraph:
  """A test fixture which enumerates a single real proto."""
  return next(random_programl_generator.EnumerateProtoTestSet())


@test.Fixture(scope="session", params=list(programl.InputOutputFormat))
def stdin_fmt(request) -> programl.InputOutputFormat:
  """A test fixture which enumerates stdin formats."""
  return request.param


@test.Fixture(scope="session", params=list(programl.InputOutputFormat))
def stdout_fmt(request) -> programl.InputOutputFormat:
  """A test fixture which enumerates stdout formats."""
  return request.param


@test.Fixture(scope="session", params=list(annotate.ANALYSES.keys()))
def analysis(request) -> programl.InputOutputFormat:
  """A test fixture which yields all analysis names."""
  return request.param


@test.Fixture(scope="session", params=(1, 3))
def n(request) -> int:
  """A test fixture enumerate values for `n`."""
  return request.param


###############################################################################
# Tests.
###############################################################################


def test_invalid_analysis(one_proto: programl_pb2.ProgramGraph, n: int):
  """Test that error is raised if the input is invalid."""
  with test.Raises(annotate.AnalysisFailed) as e_ctx:
    annotate.Annotate("invalid_analysis", one_proto, n)
  assert e_ctx.value.returncode == annotate.E_ANALYSIS_INIT


def test_invalid_input(analysis: str, n: int):
  """Test that error is raised if the input is invalid."""
  invalid_input = programl_pb2.ProgramGraph()
  with test.Raises(annotate.AnalysisFailed) as e_ctx:
    annotate.Annotate(analysis, invalid_input, n)
  assert e_ctx.value.returncode == annotate.E_INVALID_INPUT


def test_timeout(one_proto: programl_pb2.ProgramGraph):
  """Test that error is raised if the analysis times out."""
  with test.Raises(annotate.AnalysisTimeout) as e_ctx:
    annotate.Annotate("test_timeout", one_proto, timeout=1)


def test_annotate(analysis: str, real_proto: programl_pb2.ProgramGraph, n: int):
  """Test the annotator binary."""
  annotated = annotate.Annotate(analysis, real_proto, n)

  # Check that up to 'n' annotated graphs were generated.
  assert 0 <= len(annotated.graph) <= n

  # Check that output graphs have the same shape as the input graphs.
  for graph in annotated.graph:
    assert len(graph.node) == len(real_proto.node)
    assert len(graph.edge) == len(real_proto.edge)


if __name__ == "__main__":
  test.Main()
