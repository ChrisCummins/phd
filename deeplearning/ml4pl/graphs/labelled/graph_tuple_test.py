"""Unit tests for //deeplearning/ml4pl/graphs/graph_tuple."""
import networkx as nx
import numpy as np

from deeplearning.ml4pl.graphs.labelled import graph_tuple
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def graph() -> nx.MultiDiGraph:
  g = nx.MultiDiGraph()
  g.add_node("A", type="statement", x=[0, 0])
  g.add_node("B", type="statement", x=[1, 0])
  g.add_node("C", type="statement", x=[2, 0])
  g.add_node("D", type="statement", x=[3, 0])
  g.add_node("root", type="magic", x=[4, 0])
  g.add_edge("A", "B", flow="control", position=0)
  g.add_edge("B", "C", flow="control", position=0)
  g.add_edge("C", "D", flow="control", position=0)
  g.add_edge("root", "A", flow="call", position=0)
  g.add_edge("A", "D", flow="data", position=1)
  return g


def test_CreateFromNetworkX_adjacency_lists(graph: nx.MultiDiGraph):
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert d.adjacency_lists.shape == (6,)  # forward and backward edges

  # Forward edges:

  assert d.adjacency_lists[0].shape == (3, 2)  # control flow
  assert np.array_equal(d.adjacency_lists[0][0], (0, 1))  # A -> B
  assert np.array_equal(d.adjacency_lists[0][1], (1, 2))  # B -> C
  assert np.array_equal(d.adjacency_lists[0][2], (2, 3))  # C -> D

  assert d.adjacency_lists[1].shape == (1, 2)  # data flow
  assert np.array_equal(d.adjacency_lists[1][0], (0, 3))  # A -> D

  assert d.adjacency_lists[2].shape == (1, 2)  # call flow
  assert np.array_equal(d.adjacency_lists[2][0], (4, 0))  # root -> A

  # Backward edges:

  assert d.adjacency_lists[3].shape == (3, 2)  # backward control flow
  assert np.array_equal(d.adjacency_lists[3][0], (1, 0))  # A <- B
  assert np.array_equal(d.adjacency_lists[3][1], (2, 1))  # B <- C
  assert np.array_equal(d.adjacency_lists[3][2], (3, 2))  # C <- D

  assert d.adjacency_lists[4].shape == (1, 2)  # backward data flow
  assert np.array_equal(d.adjacency_lists[4][0], (3, 0))  # A <- D

  assert d.adjacency_lists[5].shape == (1, 2)  # backward call flow
  assert np.array_equal(d.adjacency_lists[5][0], (0, 4))  # root <- A


def test_CreateFromNetworkX_position_lists(graph: nx.MultiDiGraph):
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert d.edge_positions.shape == (6,)  # forward and backward edges

  # Forward edges:

  assert d.edge_positions[0].shape == (3,)  # control flow
  assert np.array_equal(d.edge_positions[0][0], 0)  # A -> B
  assert np.array_equal(d.edge_positions[0][1], 0)  # B -> C
  assert np.array_equal(d.edge_positions[0][2], 0)  # C -> D

  assert d.edge_positions[1].shape == (1,)  # data flow
  assert np.array_equal(d.edge_positions[1][0], 1)  # A -> D

  assert d.edge_positions[2].shape == (1,)  # call flow
  assert np.array_equal(d.edge_positions[2][0], 0)  # root -> A

  # Backward edges:

  assert d.edge_positions[3].shape == (3,)  # backward control flow
  assert np.array_equal(d.edge_positions[3][0], 0)  # A <- B
  assert np.array_equal(d.edge_positions[3][1], 0)  # B <- C
  assert np.array_equal(d.edge_positions[3][2], 0)  # C <- D

  assert d.edge_positions[5].shape == (1,)  # backward data flow
  assert np.array_equal(d.edge_positions[4][0], 1)  # A <- D

  assert d.edge_positions[5].shape == (1,)  # backward call flow
  assert np.array_equal(d.edge_positions[3][0], 0)  # root <- A


def test_CreateFromNetworkX_incoming_edges(graph: nx.MultiDiGraph):
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert d.incoming_edge_counts.shape == (6,)  # forward and backward edges

  # Forward edges:

  assert len(d.incoming_edge_counts[0].keys()) == 3
  assert d.incoming_edge_counts[0][1] == 1
  assert d.incoming_edge_counts[0][2] == 1
  assert d.incoming_edge_counts[0][3] == 1

  assert len(d.incoming_edge_counts[1].keys()) == 1
  assert d.incoming_edge_counts[1][3] == 1

  assert len(d.incoming_edge_counts[2].keys()) == 1
  assert d.incoming_edge_counts[2][0] == 1

  # Backward edges:

  assert len(d.incoming_edge_counts[3].keys()) == 3
  assert d.incoming_edge_counts[3][0] == 1
  assert d.incoming_edge_counts[3][1] == 1
  assert d.incoming_edge_counts[3][2] == 1

  assert len(d.incoming_edge_counts[4].keys()) == 1
  assert d.incoming_edge_counts[4][0] == 1

  assert len(d.incoming_edge_counts[5].keys()) == 1
  assert d.incoming_edge_counts[5][4] == 1


def test_CreateFromNetworkX_node_embedding_indices(graph: nx.MultiDiGraph):
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert not d.has_node_y
  assert not d.has_graph_x
  assert not d.has_graph_y

  assert d.node_x_indices.shape == (5, 2)
  assert d.node_x_indices.dtype == np.int32
  assert np.array_equal(d.node_x_indices[0], [0, 0])
  assert np.array_equal(d.node_x_indices[1], [1, 0])
  assert np.array_equal(d.node_x_indices[2], [2, 0])
  assert np.array_equal(d.node_x_indices[3], [3, 0])
  assert np.array_equal(d.node_x_indices[4], [4, 0])


def test_CreateFromNetworkX_node_labels(graph: nx.MultiDiGraph):
  graph.nodes["A"]["f"] = [4, 1, 0]
  graph.nodes["B"]["f"] = [3, 1, 0]
  graph.nodes["C"]["f"] = [2, 1, 0]
  graph.nodes["D"]["f"] = [1, 1, 0]
  graph.nodes["root"]["f"] = [0, 1, 0]
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph, node_y="f")

  assert d.has_node_y
  assert not d.has_graph_x
  assert not d.has_graph_y

  assert d.node_y.shape == (5, 3)
  assert np.array_equal(d.node_y[0], [4, 1, 0])
  assert np.array_equal(d.node_y[1], [3, 1, 0])
  assert np.array_equal(d.node_y[2], [2, 1, 0])
  assert np.array_equal(d.node_y[3], [1, 1, 0])
  assert np.array_equal(d.node_y[4], [0, 1, 0])


def test_CreateFromNetworkX_graph_features(graph: nx.MultiDiGraph):
  graph.foo = [0, 1, 2, 3]
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph, graph_x="foo")

  assert not d.has_node_y
  assert d.has_graph_x
  assert not d.has_graph_y

  assert np.array_equal(d.graph_x, [0, 1, 2, 3])


def test_CreateFromNetworkX_graph_targets(graph: nx.MultiDiGraph):
  graph.foo = [0, 1, 2, 3]
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph, graph_y="foo")

  assert not d.has_node_y
  assert not d.has_graph_x
  assert d.has_graph_y

  assert np.array_equal(d.graph_y, [0, 1, 2, 3])


def test_IncomingEdgeCountsToDense(graph: nx.MultiDiGraph):
  t = graph_tuple.GraphTuple.CreateFromNetworkX(graph)
  assert t.dense_incoming_edge_counts.shape == (5, 6)


def test_GraphTupleToNetworkx():
  g = graph_tuple.GraphTuple(
    adjacency_lists=[[(0, 1), (1, 2)], [(0, 2)]],
    incoming_edge_counts="__unused__",
    edge_positions=[[0, 0], [2]],
    node_x_indices=[1, 2, 3],
    graph_y=[1, 2, 3],
  ).ToNetworkx()

  assert g.number_of_nodes() == 3
  assert g.number_of_edges() == 3

  assert g.edges[0, 1, 0]["flow"] == "control"
  assert g.edges[1, 2, 0]["flow"] == "control"

  assert g.edges[0, 1, 0]["position"] == 0
  assert g.edges[1, 2, 0]["position"] == 0

  assert g.nodes[0]["x"] == 1
  assert g.nodes[1]["x"] == 2
  assert g.nodes[2]["x"] == 3

  assert g.y == [1, 2, 3]


if __name__ == "__main__":
  test.Main()
