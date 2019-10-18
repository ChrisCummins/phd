"""Unit tests for //deeplearning/ml4pl/graphs/labelled/reachability."""

import networkx as nx

from deeplearning.ml4pl.graphs.labelled.reachability import reachability
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def test_SetReachableNodes_distance_zero():
  g = nx.MultiDiGraph()
  g.add_node('A', type='statement')
  g.add_node('B', type='statement')
  g.add_node('C', type='statement')
  g.add_node('D', type='statement')
  g.add_edge('A', 'B', flow='control')
  g.add_edge('B', 'C', flow='control')
  g.add_edge('C', 'D', flow='control')
  distance = reachability.SetReachableNodes(g, 'D', 0)
  assert distance == 0


def test_SetReachableNodes_distance_one():
  g = nx.MultiDiGraph()
  g.add_node('A', type='statement')
  g.add_node('B', type='statement')
  g.add_node('C', type='statement')
  g.add_node('D', type='statement')
  g.add_edge('A', 'B', flow='control')
  g.add_edge('B', 'C', flow='control')
  g.add_edge('C', 'D', flow='control')
  distance = reachability.SetReachableNodes(g, 'A', 1)
  assert distance == 1


def test_SetReachableNodes_distance_two():
  g = nx.MultiDiGraph()
  g.add_node('A', type='statement')
  g.add_node('B', type='statement')
  g.add_node('C', type='statement')
  g.add_node('D', type='statement')
  g.add_edge('A', 'B', flow='control')
  g.add_edge('B', 'C', flow='control')
  g.add_edge('C', 'D', flow='control')
  distance = reachability.SetReachableNodes(g, 'A', 2)
  assert distance == 2


def test_SetReachableNodes_distance_three():
  g = nx.MultiDiGraph()
  g.add_node('A', type='statement')
  g.add_node('B', type='statement')
  g.add_node('C', type='statement')
  g.add_node('D', type='statement')
  g.add_edge('A', 'B', flow='control')
  g.add_edge('B', 'C', flow='control')
  g.add_edge('C', 'D', flow='control')
  distance = reachability.SetReachableNodes(g, 'A', 0)
  assert distance == 3


if __name__ == '__main__':
  test.Main()
