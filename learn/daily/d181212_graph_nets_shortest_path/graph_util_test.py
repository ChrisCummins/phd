"""Unit tests for //learn/daily/d181212_graph_nets_shortest_path:graph_util."""
import sys
import typing

import networkx as nx
import numpy as np
import pytest
from absl import app
from absl import flags

from learn.daily.d181212_graph_nets_shortest_path import graph_util


FLAGS = flags.FLAGS


def test_GenerateGraph_return_type():
  """Test that networkx Graph is returned."""
  graph = graph_util.GenerateGraph(
      rand=np.random.RandomState(seed=1),
      num_nodes_min_max=[10, 15], dimensions=2, theta=20, rate=1.0)
  assert isinstance(graph, nx.Graph)


def test_GenerateGraph_num_nodes():
  """Test that number of nodes is within requested range."""
  graph = graph_util.GenerateGraph(
      rand=np.random.RandomState(seed=1),
      num_nodes_min_max=[10, 15], dimensions=2, theta=20, rate=1.0)
  assert 10 <= graph.number_of_nodes() < 15


def test_GenerateGraph_num_edges():
  """Test that graph has at least one edge per node."""
  graph = graph_util.GenerateGraph(
      rand=np.random.RandomState(seed=1),
      num_nodes_min_max=[10, 15], dimensions=2, theta=20, rate=1.0)
  assert graph.number_of_edges() >= graph.number_of_nodes()


def test_GenerateGraph_is_connected():
  """Test that graph is connected."""
  graph = graph_util.GenerateGraph(
      rand=np.random.RandomState(seed=1),
      num_nodes_min_max=[10, 15], dimensions=2, theta=20, rate=1.0)
  assert nx.is_connected(graph)


def test_AddShortestPath_empty_graph():
  """Test that error raised if graph is empty."""
  g = nx.Graph()
  with pytest.raises(ValueError) as e_ctx:
    graph_util.AddShortestPath(np.random.RandomState(seed=1), g)
  assert str(e_ctx.value) == "All shortest paths are below the minimum length"


def test_AddShortestPath_min_length_too_large():
  """Test that error raised if min_length too large."""
  g = nx.Graph()
  g.add_edge('A', 'B')
  with pytest.raises(ValueError) as e_ctx:
    graph_util.AddShortestPath(np.random.RandomState(seed=1), g, min_length=5)
  assert str(e_ctx.value) == "All shortest paths are below the minimum length"


def test_AddShortestPath_result_is_digraph():
  """Test that result is a directed graph."""
  g = nx.Graph()
  g.add_edge('A', 'B')
  g.add_edge('B', 'C')
  g.add_edge('C', 'D')
  digraph = graph_util.AddShortestPath(np.random.RandomState(seed=1), g)
  assert isinstance(digraph, nx.DiGraph)


def test_AddShortestPath_num_nodes():
  """Test that result graph has same number of nodes."""
  g = nx.Graph()
  g.add_edge('A', 'B')
  g.add_edge('B', 'C')
  g.add_edge('C', 'D')
  digraph = graph_util.AddShortestPath(np.random.RandomState(seed=1), g)
  assert digraph.number_of_nodes() == 4


def test_AddShortestPath_num_edges():
  """Test that result graph has same number of edges."""
  g = nx.Graph()
  g.add_edge('A', 'B')
  g.add_edge('B', 'C')
  g.add_edge('C', 'D')
  digraph = graph_util.AddShortestPath(np.random.RandomState(seed=1), g)
  assert digraph.number_of_edges() == 6  # 3 undirected edges * 2


def test_AddShortestPath_simple_graph_path():
  """Test result path for a simple graph."""
  # Graph:
  #     A -- B -- C -- D -- E -- F
  #                 \_ G
  g = nx.Graph()
  g.add_edge('A', 'B')
  g.add_edge('B', 'C')
  g.add_edge('C', 'D')
  g.add_edge('C', 'G')
  g.add_edge('D', 'E')
  g.add_edge('E', 'F')
  # Use min_length 3 so that the path is A -> B -> C -> D
  digraph = graph_util.AddShortestPath(np.random.RandomState(seed=1), g,
                                       min_length=5)
  assert digraph.node['A']['start']
  assert digraph.node['A']['solution']
  assert not digraph.node['A']['end']

  assert not digraph.node['B']['start']
  assert digraph.node['B']['solution']
  assert not digraph.node['B']['end']

  assert not digraph.node['C']['start']
  assert digraph.node['C']['solution']
  assert not digraph.node['C']['end']

  assert not digraph.node['D']['start']
  assert digraph.node['D']['solution']
  assert not digraph.node['D']['end']

  assert not digraph.node['E']['start']
  assert digraph.node['E']['solution']
  assert not digraph.node['E']['end']

  assert not digraph.node['F']['start']
  assert digraph.node['F']['solution']
  assert digraph.node['F']['end']

  assert not digraph.node['G']['start']
  assert not digraph.node['G']['solution']
  assert not digraph.node['G']['end']


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
