"""Unit tests for //learn/daily/d181212_graph_nets_shortest_path:graph_util."""

import networkx as nx
import numpy as np
import pytest

from labm8 import app
from labm8 import test
from learn.daily.d181212_graph_nets_shortest_path import graph_util

FLAGS = app.FLAGS


def test_GenerateGraph_return_type():
  """Test that networkx Graph is returned."""
  graph = graph_util.GenerateGraph(
      rand=np.random.RandomState(seed=1),
      num_nodes_min_max=[10, 15],
      dimensions=2,
      theta=20,
      rate=1.0)
  assert isinstance(graph, nx.Graph)


def test_GenerateGraph_num_nodes():
  """Test that number of nodes is within requested range."""
  graph = graph_util.GenerateGraph(
      rand=np.random.RandomState(seed=1),
      num_nodes_min_max=[10, 15],
      dimensions=2,
      theta=20,
      rate=1.0)
  assert 10 <= graph.number_of_nodes() < 15


def test_GenerateGraph_num_edges():
  """Test that graph has at least one edge per node."""
  graph = graph_util.GenerateGraph(
      rand=np.random.RandomState(seed=1),
      num_nodes_min_max=[10, 15],
      dimensions=2,
      theta=20,
      rate=1.0)
  assert graph.number_of_edges() >= graph.number_of_nodes()


def test_GenerateGraph_is_connected():
  """Test that graph is connected."""
  graph = graph_util.GenerateGraph(
      rand=np.random.RandomState(seed=1),
      num_nodes_min_max=[10, 15],
      dimensions=2,
      theta=20,
      rate=1.0)
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
  digraph = graph_util.AddShortestPath(
      np.random.RandomState(seed=1), g, min_length=5)
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


def test_GraphToInputTarget_number_of_nodes():
  """Test number of nodes in feature graphs."""
  digraph = graph_util.AddShortestPath(
      np.random.RandomState(seed=1),
      graph_util.GenerateGraph(np.random.RandomState(seed=1), [10, 15]))
  igraph, tgraph = graph_util.GraphToInputTarget(digraph)

  assert tgraph.number_of_nodes() == digraph.number_of_nodes()
  assert igraph.number_of_nodes() == digraph.number_of_nodes()


def test_GraphToInputTarget_number_of_edges():
  """Test number of edges in feature graphs."""
  digraph = graph_util.AddShortestPath(
      np.random.RandomState(seed=1),
      graph_util.GenerateGraph(np.random.RandomState(seed=1), [10, 15]))
  igraph, tgraph = graph_util.GraphToInputTarget(digraph)

  assert igraph.number_of_edges() == digraph.number_of_edges()
  assert tgraph.number_of_edges() == digraph.number_of_edges()


def test_GraphToInputTarget_graph_features_shape():
  """Test number of features graphs."""
  digraph = graph_util.AddShortestPath(
      np.random.RandomState(seed=1),
      graph_util.GenerateGraph(np.random.RandomState(seed=1), [10, 15]))
  igraph, tgraph = graph_util.GraphToInputTarget(digraph)

  assert igraph.graph['features'].shape == (1,)
  assert tgraph.graph['features'].shape == (1,)


def test_GraphToInputTarget_node_features_shape():
  """Test number of features in nodes."""
  digraph = graph_util.AddShortestPath(
      np.random.RandomState(seed=1),
      graph_util.GenerateGraph(np.random.RandomState(seed=1), [10, 15]))
  igraph, tgraph = graph_util.GraphToInputTarget(digraph)

  for node_idx in igraph.nodes:
    assert igraph.node[node_idx]['features'].shape == (5,)
    assert tgraph.node[node_idx]['features'].shape == (2,)


def test_GraphToInputTarget_edge_features_shape():
  """Test number of features in edges."""
  digraph = graph_util.AddShortestPath(
      np.random.RandomState(seed=1),
      graph_util.GenerateGraph(np.random.RandomState(seed=1), [10, 15]))
  igraph, tgraph = graph_util.GraphToInputTarget(digraph)

  for from_idx, to_idx in igraph.edges:
    assert igraph.edges[from_idx, to_idx]['features'].shape == (1,)
    assert tgraph.edges[from_idx, to_idx]['features'].shape == (2,)


def test_GenerateGraphs_number_of_graphs():
  """Test that correct number of graphs are generated."""
  i, t, g = graph_util.GenerateGraphs(
      np.random.RandomState(seed=1), 100, [10, 12], 1000)
  assert len(i) == 100
  assert len(t) == 100
  assert len(g) == 100


if __name__ == '__main__':
  test.Main()
