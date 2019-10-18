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


def test_SetReachableNodes_distance_zero(graph):
  distance = reachability.SetReachableNodes(graph, 'D', 0)
  assert distance == 0


def test_SetReachableNodes_distance_one(graph):
  distance = reachability.SetReachableNodes(graph, 'A', 1)
  assert distance == 1


def test_SetReachableNodes_distance_two(graph):
  distance = reachability.SetReachableNodes(graph, 'A', 2)
  assert distance == 2


def test_SetReachableNodes_distance_three(graph):
  distance = reachability.SetReachableNodes(graph, 'A', 0)
  assert distance == 3


if __name__ == '__main__':
  test.Main()
