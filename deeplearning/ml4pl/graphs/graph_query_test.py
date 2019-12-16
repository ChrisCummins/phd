"""Unit tests for //deeplearning/ml4pl/graphs:graph_query."""
import networkx as nx
import pytest

from deeplearning.ml4pl.graphs import graph_query as query
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def graph():
  g = nx.MultiDiGraph()
  g.add_node("A", type="statement")
  g.add_node("B", type="statement")
  g.add_node("C", type="statement")
  g.add_node("D", type="statement")
  g.add_node("%1", type="identifier")
  g.add_node("root", type="magic")
  g.add_edge("A", "B", flow="control")
  g.add_edge("B", "C", flow="control")
  g.add_edge("C", "D", flow="control")
  g.add_edge("root", "A", flow="call")
  g.add_edge("A", "%1", flow="data")
  g.add_edge("%1", "D", flow="data")
  return g


def test_StatementNeighbors(graph):
  assert query.StatementNeighbors(graph, "A") == {"B"}


def test_StatementNeighbors_data_flow(graph):
  assert query.StatementNeighbors(graph, "A", flow="data") == {"D"}


def test_StatementIsSuccessor(graph):
  assert query.StatementIsSuccessor(graph, "A", "B")
  assert not query.StatementIsSuccessor(graph, "B", "A")


def test_StatementIsSuccessor_linear_control_path():
  g = nx.MultiDiGraph()
  g.add_edge("a", "b", type="control")
  g.add_edge("b", "c", type="control")
  assert query.StatementIsSuccessor(g, "a", "a")
  assert query.StatementIsSuccessor(g, "a", "b")
  assert query.StatementIsSuccessor(g, "a", "c")
  assert query.StatementIsSuccessor(g, "b", "c")
  assert not query.StatementIsSuccessor(g, "c", "a")
  assert not query.StatementIsSuccessor(g, "b", "a")
  assert not query.StatementIsSuccessor(g, "a", "_not_in_graph_")
  with test.Raises(Exception):
    assert not query.StatementIsSuccessor(
      g, "_not_in_graph_", "_not_in_graph2_"
    )


def test_StatementIsSuccessor_branched_control_path():
  g = nx.MultiDiGraph()
  g.add_edge("a", "b", type="control")
  g.add_edge("a", "c", type="control")
  g.add_edge("b", "d", type="control")
  g.add_edge("c", "d", type="control")
  assert query.StatementIsSuccessor(g, "a", "b")
  assert query.StatementIsSuccessor(g, "a", "c")
  assert query.StatementIsSuccessor(g, "a", "b")
  assert not query.StatementIsSuccessor(g, "b", "a")
  assert not query.StatementIsSuccessor(g, "b", "c")
  assert query.StatementIsSuccessor(g, "b", "d")


def test_GetStatementsForNode_node():
  """Test the nodes returned when root is a statementt."""
  g = nx.MultiDiGraph()
  g.add_node("foo", type="statement")

  nodes = list(query.GetStatementsForNode(g, "foo"))
  assert nodes == ["foo"]


def test_GetStatementsForNode_identifier():
  """Test the nodes returned when root is an identifier."""
  g = nx.MultiDiGraph()
  g.add_node("foo", type="statement")
  g.add_node("bar", type="statement")
  g.add_node("%1", type="identifier")
  g.add_edge("foo", "%1", flow="data")
  g.add_edge("%1", "bar", flow="data")

  nodes = list(query.GetStatementsForNode(g, "%1"))
  assert nodes == ["foo", "bar"]


if __name__ == "__main__":
  test.Main()
