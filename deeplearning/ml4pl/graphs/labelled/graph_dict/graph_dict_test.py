"""Unit tests for //deeplearning/ml4pl/graphs/labelled/graph_dict."""
import networkx as nx
import numpy as np
import pytest

from deeplearning.ml4pl.graphs.labelled.graph_dict import graph_dict
from labm8 import app
from labm8 import test


FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def graph() -> nx.MultiDiGraph:
  g = nx.MultiDiGraph()
  g.add_node('A', type='statement')
  g.add_node('B', type='statement')
  g.add_node('C', type='statement')
  g.add_node('D', type='statement')
  g.add_node('root', type='magic')
  g.add_edge('A', 'B', flow='control')
  g.add_edge('B', 'C', flow='control')
  g.add_edge('C', 'D', flow='control')
  g.add_edge('root', 'A', flow='call')
  g.add_edge('A', 'D', flow='data')
  return g


def test_ToGraphDict_adjacency_lists(graph: nx.MultiDiGraph):
  d = graph_dict.ToGraphDict(graph, {'call', 'control', 'data'})

  assert len(d['adjacency_lists']) == 3

  print(d['adjacency_lists'])

  assert len(d['adjacency_lists'][0]) == 1  # call flow
  assert np.array_equal(d['adjacency_lists'][0][0], (4, 0))  # root -> A

  assert len(d['adjacency_lists'][1]) == 3  # control flow
  assert np.array_equal(d['adjacency_lists'][1][0], (0, 1))  # A -> B
  assert np.array_equal(d['adjacency_lists'][1][1], (1, 2))  # B -> C
  assert np.array_equal(d['adjacency_lists'][1][2], (2, 3))  # C -> D

  assert len(d['adjacency_lists'][2]) == 1  # data flow
  assert np.array_equal(d['adjacency_lists'][2][0], (0, 3))  # A -> D


def test_ToGraphDict_incoming_edges(graph: nx.MultiDiGraph):
  d = graph_dict.ToGraphDict(graph, {'call', 'control', 'data'})

  print(d['incoming_edge_counts'])

  assert len(d['incoming_edge_counts']) == 3

  assert len(d['incoming_edge_counts'][0]) == 1
  assert d['incoming_edge_counts'][0][0] == 1

  assert len(d['incoming_edge_counts'][1]) == 3
  assert d['incoming_edge_counts'][1][1] == 1
  assert d['incoming_edge_counts'][1][2] == 1
  assert d['incoming_edge_counts'][1][3] == 1

  assert len(d['incoming_edge_counts'][2]) == 1
  assert d['incoming_edge_counts'][2][3] == 1


def test_ToGraphDict_node_features(graph: nx.MultiDiGraph):
  graph.nodes['A']['x'] = [0]
  graph.nodes['B']['x'] = [1]
  graph.nodes['C']['x'] = [2]
  graph.nodes['D']['x'] = [3]
  graph.nodes['root']['x'] = [4]
  d = graph_dict.ToGraphDict(graph, {'call', 'control', 'data'})

  assert 'node_x' in d
  assert 'node_y' not in d
  assert 'edge_x' not in d
  assert 'edge_y' not in d
  assert 'graph_x' not in d
  assert 'graph_y' not in d

  assert len(d['node_x']) == 5
  assert d['node_x'][0] == [0]
  assert d['node_x'][1] == [1]
  assert d['node_x'][2] == [2]
  assert d['node_x'][3] == [3]
  assert d['node_x'][4] == [4]


def test_ToGraphDict_node_labels(graph: nx.MultiDiGraph):
  graph.nodes['A']['f'] = [4]
  graph.nodes['B']['f'] = [3]
  graph.nodes['C']['f'] = [2]
  graph.nodes['D']['f'] = [1]
  graph.nodes['root']['f'] = [0]
  d = graph_dict.ToGraphDict(graph, {'call', 'control', 'data'}, node_y='f')

  assert 'node_x' not in d
  assert 'node_y' in d
  assert 'edge_x' not in d
  assert 'edge_y' not in d
  assert 'graph_x' not in d
  assert 'graph_y' not in d

  assert len(d['node_y']) == 5
  assert d['node_y'][0] == [4]
  assert d['node_y'][1] == [3]
  assert d['node_y'][2] == [2]
  assert d['node_y'][3] == [1]
  assert d['node_y'][4] == [0]


def test_ToGraphDict_edge_features(graph: nx.MultiDiGraph):
  graph.edges['A', 'B', 0]['x'] = [0, 1, 1, 1]
  graph.edges['B', 'C', 0]['x'] = [1, 0, 1, 1]
  graph.edges['C', 'D', 0]['x'] = [1, 1, 0, 1]
  graph.edges['root', 'A', 0]['x'] = [1, 1, 1, 0]
  graph.edges['A', 'D', 0]['x'] = [1, 1, 1, 1]

  d = graph_dict.ToGraphDict(graph, {'call', 'control', 'data'})

  assert 'node_x' not in d
  assert 'node_y' not in d
  assert 'edge_x' in d
  assert 'edge_y' not in d
  assert 'graph_x' not in d
  assert 'graph_y' not in d

  assert len(d['edge_x']) == 3

  assert len(d['edge_x'][0]) == 1  # call flow
  assert np.array_equal(d['edge_x'][0][0], [1, 1, 1, 0])  # root -> A

  assert len(d['edge_x'][1]) == 3  # control flow
  assert np.array_equal(d['edge_x'][1][0], [0, 1, 1, 1])  # A -> B
  assert np.array_equal(d['edge_x'][1][1], [1, 0, 1, 1])  # B -> C
  assert np.array_equal(d['edge_x'][1][2], [1, 1, 0, 1])  # C -> D

  assert len(d['edge_x'][2]) == 1  # data flow
  assert np.array_equal(d['edge_x'][2][0], [1, 1, 1, 1])  # A -> D


def test_ToGraphDict_edge_targets(graph: nx.MultiDiGraph):
  graph.edges['A', 'B', 0]['y'] = [3]
  graph.edges['B', 'C', 0]['y'] = [4]
  graph.edges['C', 'D', 0]['y'] = [5]
  graph.edges['root', 'A', 0]['y'] = [6]
  graph.edges['A', 'D', 0]['y'] = [7]

  d = graph_dict.ToGraphDict(graph, {'call', 'control', 'data'})

  assert 'node_x' not in d
  assert 'node_y' not in d
  assert 'edge_x' not in d
  assert 'edge_y' in d
  assert 'graph_x' not in d
  assert 'graph_y' not in d

  assert len(d['edge_y']) == 3

  assert len(d['edge_y'][0]) == 1  # call flow
  assert np.array_equal(d['edge_y'][0][0], [6])  # root -> A

  assert len(d['edge_y'][1]) == 3  # control flow
  assert np.array_equal(d['edge_y'][1][0], [3])  # A -> B
  assert np.array_equal(d['edge_y'][1][1], [4])  # B -> C
  assert np.array_equal(d['edge_y'][1][2], [5])  # C -> D

  assert len(d['edge_y'][2]) == 1  # data flow
  assert np.array_equal(d['edge_y'][2][0], [7])  # A -> D


def test_ToGraphDict_graph_features(graph: nx.MultiDiGraph):
  graph.foo = [0, 1, 2, 3]
  d = graph_dict.ToGraphDict(graph, {'call', 'control', 'data'}, graph_x='foo')

  assert 'node_x' not in d
  assert 'node_y' not in d
  assert 'edge_x' not in d
  assert 'edge_y' not in d
  assert 'graph_x' in d
  assert 'graph_y' not in d

  assert np.array_equal(d['graph_x'], [0, 1, 2, 3])


def test_ToGraphDict_graph_targets(graph: nx.MultiDiGraph):
  graph.foo = [0, 1, 2, 3]
  d = graph_dict.ToGraphDict(graph, {'call', 'control', 'data'}, graph_y='foo')

  assert 'node_x' not in d
  assert 'node_y' not in d
  assert 'edge_x' not in d
  assert 'edge_y' not in d
  assert 'graph_x' not in d
  assert 'graph_y' in d

  assert np.array_equal(d['graph_y'], [0, 1, 2, 3])


def test_IncomingEdgeCountsToDense():
  incoming_edge_counts = [{
    0: 1
  }, {
    0: 10,
    3: 2
  }, {
    0: 6,
    2: 1
  }]
  dense = graph_dict.IncomingEdgeCountsToDense(
      incoming_edge_counts, node_count=4, edge_type_count=3)

  assert np.array_equal(
      dense, np.array([
          [1, 10, 6],
          [0, 0, 0],
          [0, 0, 1],
          [0, 2, 0],
      ]))

def test_GraphDictToNetworkx():
  g = graph_dict.GraphDictToNetworkx({
    'adjacency_lists': [[(0, 1), (1, 2)], [(0, 2)]],
    'node_x': [[1], [2], [3]],
    'edge_y': [[[1], [2]], [[3]]],
  })

  assert g.number_of_nodes() == 3
  assert g.number_of_edges() == 3

  assert g.nodes[0]['x'] == [1]
  assert g.nodes[1]['x'] == [2]
  assert g.nodes[2]['x'] == [3]

  assert g.edges[0, 1, 0]['y'] == [1]
  assert g.edges[1, 2, 0]['y'] == [2]
  assert g.edges[0, 2, 0]['y'] == [3]


if __name__ == '__main__':
  test.Main()
