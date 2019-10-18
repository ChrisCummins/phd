"""Unit tests for //deeplearning/ml4pl/graphs:graph_iterators."""
import networkx as nx
import pytest

from deeplearning.ml4pl.graphs import graph_iterators as iterators
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


def test_StatementNodeIterator(graph):
  assert len(list(iterators.StatementNodeIterator(graph))) == 4


def test_IdentifierNodeIterator(graph):
  assert len(list(iterators.IdentifierNodeIterator(graph))) == 1


def test_EntryBlockIterator(graph):
  assert len(list(iterators.EntryBlockIterator(graph))) == 1


def test_ExitBlockIterator(graph):
  assert len(list(iterators.ExitBlockIterator(graph))) == 1


def test_ControlFlowEdgeIterator(graph):
  assert len(list(iterators.ControlFlowEdgeIterator(graph))) == 3


def test_DataFlowEdgeIterator(graph):
  assert len(list(iterators.DataFlowEdgeIterator(graph))) == 2


if __name__ == '__main__':
  test.Main()
