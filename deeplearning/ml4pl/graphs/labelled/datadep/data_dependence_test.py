"""Unit tests for //deeplearning/ml4pl/graphs/labelled/datadep:data_dependence."""
import networkx as nx

from deeplearning.ml4pl.graphs.labelled.datadep import data_dependence
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def test_AnnotateDominatorTree():
  g = nx.MultiDiGraph()
  g.add_node('A', type='statement')
  g.add_node('B', type='statement')
  g.add_node('C', type='statement')
  g.add_node('D', type='statement')
  g.add_node('E', type='statement')

  g.add_edge('A', 'B', flow='control')
  g.add_edge('A', 'C', flow='control')
  g.add_edge('B', 'D', flow='control')
  g.add_edge('C', 'D', flow='control')
  g.add_edge('A', 'E', flow='control')

  g.add_edge('A', 'B', flow='data')
  g.add_edge('A', 'C', flow='data')
  g.add_edge('C', 'D', flow='data')

  dependent_node_count, max_steps = data_dependence.AnnotateDataDependencies(
      g, 'D')
  assert dependent_node_count == 3
  assert max_steps == 3

  # Features
  assert not g.nodes['A']['x']
  assert not g.nodes['B']['x']
  assert not g.nodes['C']['x']
  assert g.nodes['D']['x']
  assert not g.nodes['E']['x']

  # Labels
  assert g.nodes['A']['y']
  assert not g.nodes['B']['y']
  assert g.nodes['C']['y']
  assert g.nodes['D']['y']
  assert not g.nodes['E']['y']


if __name__ == '__main__':
  test.Main()