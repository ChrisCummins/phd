"""Unit tests for //deeplearning/ml4pl/graphs/labelled/reachability."""
import networkx as nx
import pytest

from deeplearning.ml4pl.graphs.labelled.reachability import reachability
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@pytest.fixture(scope="function")
def graph():
  g = nx.MultiDiGraph()
  g.add_node("A", type="statement", x=-1)
  g.add_node("B", type="statement", x=-1)
  g.add_node("C", type="statement", x=-1)
  g.add_node("D", type="statement", x=-1)
  g.add_edge("A", "B", flow="control")
  g.add_edge("B", "C", flow="control")
  g.add_edge("C", "D", flow="control")
  return g


def test_SetReachableNodes_reachable_node_count_D(graph):
  reachable_node_count, _ = reachability.SetReachableNodes(graph, "D")
  assert reachable_node_count == 1


def test_SetReachableNodes_reachable_node_count_A(graph):
  reachable_node_count, _ = reachability.SetReachableNodes(graph, "A")
  assert reachable_node_count == 4


def test_SetReachableNodes_data_flow_steps_D(graph):
  _, data_flow_steps = reachability.SetReachableNodes(graph, "D")
  assert data_flow_steps == 1


def test_SetReachableNodes_data_flow_steps_A(graph):
  _, data_flow_steps = reachability.SetReachableNodes(graph, "A")
  assert data_flow_steps == 4


def test_SetReachableNodes_node_x(graph):
  _ = reachability.SetReachableNodes(graph, "A")
  assert graph.nodes["A"]["x"] == [-1, 1]
  assert graph.nodes["B"]["x"] == [-1, 0]
  assert graph.nodes["C"]["x"] == [-1, 0]
  assert graph.nodes["D"]["x"] == [-1, 0]


def test_SetReachableNodes_node_y(graph):
  _ = reachability.SetReachableNodes(graph, "B")
  assert not graph.nodes["A"]["y"]
  assert graph.nodes["B"]["y"]
  assert graph.nodes["C"]["y"]
  assert graph.nodes["D"]["y"]


if __name__ == "__main__":
  test.Main()
