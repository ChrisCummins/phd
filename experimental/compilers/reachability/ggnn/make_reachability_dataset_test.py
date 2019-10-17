"""Unit tests for //experimental/compilers/reachability/ggnn:make_reachability_dataset."""

import networkx as nx

from experimental.compilers.reachability.ggnn import \
  make_reachability_dataset as mrd
from labm8 import app
from labm8 import test


FLAGS = app.FLAGS


def test_SetReachableNodes_distance():
  g = nx.MultiDiGraph()
  g.add_node('A', type='statement')
  g.add_node('B', type='statement')
  g.add_node('C', type='statement')
  g.add_node('D', type='statement')
  g.add_edge('A', 'B', flow='control')
  g.add_edge('B', 'C', flow='control')
  g.add_edge('C', 'D', flow='control')
  distance = mrd.SetReachableNodes(g, 'A', 0)
  assert distance == 3


def test_SetReachableNodes_distance_one():
  g = nx.MultiDiGraph()
  g.add_node('A', type='statement')
  g.add_node('B', type='statement')
  g.add_node('C', type='statement')
  g.add_node('D', type='statement')
  g.add_edge('A', 'B', flow='control')
  g.add_edge('B', 'C', flow='control')
  g.add_edge('C', 'D', flow='control')
  distance = mrd.SetReachableNodes(g, 'A', 1)
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
  distance = mrd.SetReachableNodes(g, 'A', 2)
  assert distance == 2


if __name__ == '__main__':
  test.Main()
