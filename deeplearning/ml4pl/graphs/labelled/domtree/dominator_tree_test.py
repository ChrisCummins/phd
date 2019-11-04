"""Unit tests for //deeplearning/ml4pl/graphs/labelled/domtree:dominator_tree."""
import networkx as nx
from labm8 import app
from labm8 import test

from deeplearning.ml4pl.graphs.labelled.domtree import dominator_tree

FLAGS = app.FLAGS


def test_AnnotateDominatorTree():
  g = nx.MultiDiGraph()
  g.add_node('A', type='statement')
  g.add_node('B', type='statement')
  g.add_node('C', type='statement')
  g.add_node('D', type='statement')
  g.add_node('E', type='statement')
  g.add_node('%1', type='identifier')
  g.add_edge('A', 'B', flow='control')
  g.add_edge('A', 'C', flow='control')
  g.add_edge('B', 'D', flow='control')
  g.add_edge('C', 'D', flow='control')
  g.add_edge('%1', 'C', flow='data')
  g.add_edge('A', 'C', flow='data')
  g.add_edge('E', 'B', flow='control')

  dominated_node_count, max_steps = dominator_tree.AnnotateDominatorTree(g, 'A')
  assert dominated_node_count == 2
  # TODO(cec): What is the correct value here?
  assert max_steps in {2, 3}

  # Features
  assert g.nodes['A']['x'] == 1
  assert g.nodes['B']['x'] == 0
  assert g.nodes['C']['x'] == 0
  assert g.nodes['D']['x'] == 0
  assert g.nodes['E']['x'] == 0
  assert g.nodes['%1']['x'] == 0

  # Labels
  assert g.nodes['A']['y']
  assert not g.nodes['B']['y']
  assert g.nodes['C']['y']
  assert not g.nodes['D']['y']
  assert not g.nodes['E']['y']
  assert not g.nodes['%1']['y']


if __name__ == '__main__':
  test.Main()
