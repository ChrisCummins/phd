"""Unit tests for //deeplearning/ml4pl/graphs:graph_query."""
import networkx as nx
import pytest

from deeplearning.ml4pl.graphs import graph_query as query
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
  g.add_node('%1', type='identifier')
  g.add_node('root', type='magic')
  g.add_edge('A', 'B', flow='control')
  g.add_edge('B', 'C', flow='control')
  g.add_edge('C', 'D', flow='control')
  g.add_edge('root', 'A', flow='call')
  g.add_edge('A', '%1', flow='data')
  g.add_edge('%1', 'D', flow='data')
  return g


def test_StatementNeighbors(graph):
  assert query.StatementNeighbors(graph, 'A') == {'B'}


def test_StatementNeighbors_data_flow(graph):
  assert query.StatementNeighbors(graph, 'A', flow='data') == {'D'}


def test_StatementIsSuccessor(graph):
  assert query.StatementIsSuccessor(graph, 'A', 'B')
  assert not query.StatementIsSuccessor(graph, 'B', 'A')


if __name__ == '__main__':
  test.Main()