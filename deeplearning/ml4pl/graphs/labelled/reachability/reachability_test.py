"""Unit tests for //deeplearning/ml4pl/graphs/labelled/reachability."""
import networkx as nx
import pytest

from deeplearning.ml4pl.graphs.labelled.reachability import reachability
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def graph():
  g = nx.MultiDiGraph()
  g.add_node('A', type='statement')
  g.add_node('B', type='statement')
  g.add_node('C', type='statement')
  g.add_node('D', type='statement')
  g.add_edge('A', 'B', flow='control')
  g.add_edge('B', 'C', flow='control')
  g.add_edge('C', 'D', flow='control')
  return g


def test_SetReachableNodes_reachable_node_count_zero(graph):
  reachable_node_count, _ = reachability.SetReachableNodes(graph, 'D', 0)
  assert reachable_node_count == 1


def test_SetReachableNodes_reachable_node_count_one(graph):
  reachable_node_count, _ = reachability.SetReachableNodes(graph, 'A', 1)
  assert reachable_node_count == 1


def test_SetReachableNodes_reachable_node_count_two(graph):
  reachable_node_count, _ = reachability.SetReachableNodes(graph, 'A', 2)
  assert reachable_node_count == 2


def test_SetReachableNodes_reachable_node_count_three(graph):
  reachable_node_count, _ = reachability.SetReachableNodes(graph, 'A', 0)
  assert reachable_node_count == 4


def test_SetReachableNodes_data_flow_steps_zero(graph):
  _, data_flow_steps = reachability.SetReachableNodes(graph, 'D', 0)
  assert data_flow_steps == 1


def test_SetReachableNodes_data_flow_steps_one(graph):
  _, data_flow_steps = reachability.SetReachableNodes(graph, 'A', 1)
  assert data_flow_steps == 1


def test_SetReachableNodes_data_flow_steps_two(graph):
  _, data_flow_steps = reachability.SetReachableNodes(graph, 'A', 2)
  assert data_flow_steps == 2


def test_SetReachableNodes_data_flow_steps_three(graph):
  _, data_flow_steps = reachability.SetReachableNodes(graph, 'A', 0)
  assert data_flow_steps == 4


def test_SetReachableNodes_node_x(graph):
  _ = reachability.SetReachableNodes(graph, 'A', 2)
  assert graph.nodes['A']['x']
  assert not graph.nodes['B']['x']
  assert not graph.nodes['C']['x']
  assert not graph.nodes['D']['x']


def test_SetReachableNodes_node_y(graph):
  _ = reachability.SetReachableNodes(graph, 'A', 2)
  assert graph.nodes['A']['y']
  assert graph.nodes['B']['y']
  assert not graph.nodes['C']['y']
  assert not graph.nodes['D']['y']


if __name__ == '__main__':
  test.Main()
