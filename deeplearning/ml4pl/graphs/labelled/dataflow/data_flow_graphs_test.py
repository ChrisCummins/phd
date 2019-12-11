"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow/reachability."""
from typing import List
from typing import Optional

import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from labm8.py import test

FLAGS = test.FLAGS


class MockNetworkXDataFlowGraphAnnotator(
  data_flow_graphs.NetworkXDataFlowGraphAnnotator
):
  """A mock networkx annotator for testing."""

  def __init__(self, node_y: Optional[List[int]] = None):
    self.node_y = node_y

  def RootNodeType(self) -> programl_pb2.Node.Type:
    """The root node type."""
    return programl_pb2.Node.STATEMENT

  def Annotate(self, g: nx.MultiDiGraph, root_node: int) -> nx.MultiDiGraph:
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

    g.graph["data_flow_root_node"] = root_node

    return g


def test_MakeAnnotated_no_root_nodes():
  """Test that a graph with no root nodes produces no annotations."""
  annotator = MockNetworkXDataFlowGraphAnnotator()
  builder = programl.GraphBuilder()
  annotated = annotator.MakeAnnotated(builder.proto)
  assert len(annotated.graphs) == 0
  assert len(annotated.protos) == 0


def test_MakeAnnotated_no_node_y():
  """Test annotator with no node labels."""
  annotator = MockNetworkXDataFlowGraphAnnotator()
  builder = programl.GraphBuilder()
  builder.AddNode()
  annotated = annotator.MakeAnnotated(builder.proto)
  assert len(annotated.graphs) == 1
  assert len(annotated.protos) == 1
  assert annotated.graphs[0].nodes[0]["y"] == []
  assert annotated.protos[0].node[0].y == []


def test_MakeAnnotated_node_y():
  """Test annotator with node labels."""
  annotator = MockNetworkXDataFlowGraphAnnotator(node_y=[1, 2])
  builder = programl.GraphBuilder()
  builder.AddNode()
  annotated = annotator.MakeAnnotated(builder.proto)
  assert len(annotated.graphs) == 1
  assert len(annotated.protos) == 1
  assert annotated.graphs[0].nodes[0]["y"] == [1, 2]
  assert annotated.protos[0].node[0].y == [1, 2]


def test_MakeAnnotated_node_x():
  """Test that node X values get appended."""
  annotator = MockNetworkXDataFlowGraphAnnotator()
  builder = programl.GraphBuilder()
  for _ in range(10):
    builder.AddNode()
  annotated = annotator.MakeAnnotated(builder.proto)
  assert len(annotated.graphs) == 10
  assert len(annotated.protos) == 10
  # Check each graph to ensure that each graph gets a fresh set of node x
  # arrays to append to, else these lists will graph in length.
  for graph, proto in zip(annotated.graphs, annotated.protos):
    assert graph.nodes[0]["x"] == [0]
    assert graph.nodes[1]["x"] == [1]
    assert graph.nodes[2]["x"] == [2]

    assert proto.node[0].x == [0]
    assert proto.node[1].x == [1]
    assert proto.node[2].x == [2]


def test_MakeAnnotated_no_graph_limit():
  """Test annotator with node labels."""
  annotator = MockNetworkXDataFlowGraphAnnotator(node_y=[1, 2])
  builder = programl.GraphBuilder()
  builder.AddNode()
  builder.AddNode()
  builder.AddNode()
  annotated = annotator.MakeAnnotated(builder.proto)
  assert len(annotated.graphs) == 3
  assert len(annotated.protos) == 3


@test.Parametrize("n", (1, 10, 100))
def test_MakeAnnotated_graph_limit(n: int):
  """Test annotator with node labels."""
  annotator = MockNetworkXDataFlowGraphAnnotator(node_y=[1, 2])
  builder = programl.GraphBuilder()
  for _ in range(100):
    builder.AddNode()
  annotated = annotator.MakeAnnotated(builder.proto, n=n)
  assert len(annotated.graphs) == n
  assert len(annotated.protos) == n


def test_MakeAnnotated_graph_subset_of_root_nodes():
  """Test the number of graphs when only a subset of nodes are roots."""
  annotator = MockNetworkXDataFlowGraphAnnotator(node_y=[1, 2])
  builder = programl.GraphBuilder()
  for i in range(50):
    if i % 2 == 0:
      builder.AddNode(type=programl_pb2.Node.STATEMENT)
    else:
      builder.AddNode(type=programl_pb2.Node.IDENTIFIER)
  annotated = annotator.MakeAnnotated(builder.proto)
  # Only the 25 statement nodes are used as roots.
  assert len(annotated.graphs) == 25
  assert len(annotated.protos) == 25


if __name__ == "__main__":
  test.Main()
