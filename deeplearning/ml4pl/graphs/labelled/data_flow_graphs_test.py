"""Unit tests for //deeplearning/ml4pl/graphs/labelled/reachability."""
import networkx as nx

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled import data_flow_graphs
from labm8.py import test

FLAGS = test.FLAGS


class MockAnnotator(data_flow_graphs.DataFlowGraphAnnotator):
  """A mock annotator for testing."""

  def RootNodeType(self) -> programl_pb2.Node.Type:
    """The root node type."""
    return programl_pb2.Node.STATEMENT

  def Annotate(
    self, g: nx.MultiDiGraph, root_node: int
  ) -> data_flow_graphs.DataFlowAnnotatedGraph:
    """Produce annotations."""
    return data_flow_graphs.DataFlowAnnotatedGraph(g=g, root_node=root_node)


@test.Fixture(scope="function")
def empty_graph() -> nx.MultiDiGraph:
  """A test fixture which returns an empty graph."""
  return nx.MultiDiGraph()


@test.Fixture(scope="function")
def annotator() -> MockAnnotator:
  """A test fixture that returns a mock data flow annotator."""
  return MockAnnotator()


def test_MakeAnnotated_no_root_nodes(
  empty_graph: nx.MultiDiGraph, annotator: MockAnnotator
):
  """Test that a graph with no root nodes produces no annotations."""
  annotated = list(annotator.MakeAnnotated(empty_graph))
  assert len(annotated) == 0


if __name__ == "__main__":
  test.Main()
