"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow/reachability."""
from typing import List
from typing import Optional

import networkx as nx

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from labm8.py import test

FLAGS = test.FLAGS


class MockAnnotator(data_flow_graphs.DataFlowGraphAnnotator):
  """A mock annotator for testing."""

  def __init__(self, node_y: Optional[List[int]] = None):
    self.node_y = node_y

  def RootNodeType(self) -> programl_pb2.Node.Type:
    """The root node type."""
    return programl_pb2.Node.STATEMENT

  def Annotate(
    self, g: nx.MultiDiGraph, root_node: int
  ) -> data_flow_graphs.DataFlowAnnotatedGraph:
    """Produce annotations.

    If node_y was passed to constructor, then that value is set. The node x
    lists are concated with the node ID.
    """
    # Add a node x feature.
    for node, data in g.nodes(data=True):
      data["x"].append(node)

    # Set a node y vector if desired.
    if self.node_y:
      for _, data in g.nodes(data=True):
        data["y"] = self.node_y

    return data_flow_graphs.DataFlowAnnotatedGraph(g=g, root_node=root_node)


def test_MakeAnnotated_no_root_nodes():
  """Test that a graph with no root nodes produces no annotations."""
  annotator = MockAnnotator()
  g = nx.MultiDiGraph()
  annotated = list(annotator.MakeAnnotated(g))
  assert len(annotated) == 0


def test_MakeAnnotated_no_node_y():
  """Test annotator with no node labels."""
  annotator = MockAnnotator()
  g = nx.MultiDiGraph()
  g.add_node(0, type=programl_pb2.Node.STATEMENT, x=[], y=[])
  annotated = list(annotator.MakeAnnotated(g))
  assert len(annotated) == 1
  assert annotated[0].g.nodes[0]["y"] == []


def test_MakeAnnotated_node_y():
  """Test annotator with node labels."""
  annotator = MockAnnotator(node_y=[1, 2])
  g = nx.MultiDiGraph()
  g.add_node(0, type=programl_pb2.Node.STATEMENT, x=[], y=[])
  annotated = list(annotator.MakeAnnotated(g))
  assert len(annotated) == 1
  assert annotated[0].g.nodes[0]["y"] == [1, 2]


def test_MakeAnnotated_node_x():
  """Test that node X values get appended."""
  annotator = MockAnnotator()
  g = nx.MultiDiGraph()
  g.add_node(0, type=programl_pb2.Node.STATEMENT, x=[], y=[])
  g.add_node(1, type=programl_pb2.Node.STATEMENT, x=[], y=[])
  g.add_node(2, type=programl_pb2.Node.STATEMENT, x=[], y=[])
  annotated = list(annotator.MakeAnnotated(g, n=2))
  assert len(annotated) == 2
  # Check the first graph.
  assert annotated[0].g.nodes[0]["x"] == [0]
  assert annotated[0].g.nodes[1]["x"] == [1]
  assert annotated[0].g.nodes[2]["x"] == [2]
  # Check the second graph. This is to ensure that each graph gets a fresh set
  # of node x arrays to append to, else these lists will graph in length.
  assert annotated[1].g.nodes[0]["x"] == [0]
  assert annotated[1].g.nodes[1]["x"] == [1]
  assert annotated[1].g.nodes[2]["x"] == [2]


def test_MakeAnnotated_no_graph_limit():
  """Test annotator with node labels."""
  annotator = MockAnnotator(node_y=[1, 2])
  g = nx.MultiDiGraph()
  g.add_node(0, type=programl_pb2.Node.STATEMENT, x=[], y=[])
  g.add_node(1, type=programl_pb2.Node.STATEMENT, x=[], y=[])
  g.add_node(2, type=programl_pb2.Node.IDENTIFIER, x=[], y=[])
  annotated = list(annotator.MakeAnnotated(g))
  assert len(annotated) == 2


def test_MakeAnnotated_graph_limit():
  """Test annotator with node labels."""
  annotator = MockAnnotator(node_y=[1, 2])
  g = nx.MultiDiGraph()
  g.add_node(0, type=programl_pb2.Node.STATEMENT, x=[], y=[])
  g.add_node(1, type=programl_pb2.Node.STATEMENT, x=[], y=[])
  g.add_node(2, type=programl_pb2.Node.IDENTIFIER, x=[], y=[])
  annotated = list(annotator.MakeAnnotated(g, n=1))
  assert len(annotated) == 1


if __name__ == "__main__":
  test.Main()
