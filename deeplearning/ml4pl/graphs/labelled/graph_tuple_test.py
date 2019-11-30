"""Unit tests for //deeplearning/ml4pl/graphs/graph_tuple."""
import networkx as nx
import numpy as np

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled import graph_tuple
from deeplearning.ml4pl.graphs.migrate import networkx_to_protos
from deeplearning.ml4pl.graphs.unlabelled.cdfg import random_cdfg_generator
from labm8.py import app
from labm8.py import decorators
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def graph() -> nx.MultiDiGraph:
  g = nx.MultiDiGraph()
  g.add_node(0, type=programl_pb2.Node.STATEMENT, discrete_x=[4, 0])
  g.add_node(1, type=programl_pb2.Node.STATEMENT, discrete_x=[0, 0])
  g.add_node(2, type=programl_pb2.Node.STATEMENT, discrete_x=[1, 0])
  g.add_node(3, type=programl_pb2.Node.STATEMENT, discrete_x=[2, 1])
  g.add_node(4, type=programl_pb2.Node.IDENTIFIER, discrete_x=[3, 0])
  g.add_edge(0, 1, flow=programl_pb2.Edge.CALL, position=0)
  g.add_edge(1, 2, flow=programl_pb2.Edge.CONTROL, position=0)
  g.add_edge(2, 3, flow=programl_pb2.Edge.CONTROL, position=0)
  g.add_edge(4, 3, flow=programl_pb2.Edge.DATA, position=0)
  g.graph["discrete_x"] = []
  g.graph["discrete_y"] = []
  return g


def test_CreateFromNetworkX_adjacency_lists(graph: nx.MultiDiGraph):
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert d.adjacency_lists.shape == (3,)

  assert d.adjacency_lists[programl_pb2.Edge.CONTROL].dtype == np.int32
  assert d.adjacency_lists[programl_pb2.Edge.DATA].dtype == np.int32
  assert d.adjacency_lists[programl_pb2.Edge.CALL].dtype == np.int32

  assert np.array_equal(
    d.adjacency_lists[programl_pb2.Edge.CONTROL], np.array([(1, 2), (2, 3)])
  )
  assert np.array_equal(
    d.adjacency_lists[programl_pb2.Edge.DATA], np.array([(4, 3)])
  )
  assert np.array_equal(
    d.adjacency_lists[programl_pb2.Edge.CALL], np.array([(0, 1)])
  )


def test_CreateFromNetworkX_edge_positions(graph: nx.MultiDiGraph):
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert d.edge_positions.shape == (3,)

  assert d.edge_positions[programl_pb2.Edge.CONTROL].dtype == np.int32
  assert d.edge_positions[programl_pb2.Edge.DATA].dtype == np.int32
  assert d.edge_positions[programl_pb2.Edge.CALL].dtype == np.int32

  assert np.array_equal(
    d.edge_positions[programl_pb2.Edge.CONTROL], np.array([0, 0])
  )
  assert np.array_equal(d.edge_positions[programl_pb2.Edge.DATA], np.array([0]))
  assert np.array_equal(d.edge_positions[programl_pb2.Edge.CALL], np.array([0]))


def test_CreateFromNetworkX_node_x(graph: nx.MultiDiGraph):
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert not d.has_node_y
  assert not d.has_graph_x
  assert not d.has_graph_y

  assert d.node_x.dtype == np.int32
  assert np.array_equal(
    d.node_x, np.array([(4, 0), (0, 0), (1, 0), (2, 1), (3, 0),])
  )


def test_CreateFromNetworkX_node_y(graph: nx.MultiDiGraph):
  graph.nodes[0]["real_y"] = [4, 1, 0]
  graph.nodes[1]["real_y"] = [3, 1, 0]
  graph.nodes[2]["real_y"] = [2, 1, 0]
  graph.nodes[3]["real_y"] = [1, 1, 0]
  graph.nodes[4]["real_y"] = [0, 1, 0]
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert d.has_node_y
  assert not d.has_graph_x
  assert not d.has_graph_y

  assert d.node_y.shape == (5, 3)
  assert d.node_y.dtype == np.float32
  assert np.array_equal(
    d.node_y, np.array([(4, 1, 0), (3, 1, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0),])
  )


def test_CreateFromNetworkX_graph_x(graph: nx.MultiDiGraph):
  graph.graph["discrete_x"] = [0, 1, 2, 3]
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert not d.has_node_y
  assert d.has_graph_x
  assert not d.has_graph_y

  assert d.graph_x.dtype == np.int32
  assert np.array_equal(d.graph_x, np.array([0, 1, 2, 3]))


def test_CreateFromNetworkX_graph_y(graph: nx.MultiDiGraph):
  graph.graph["discrete_y"] = [0, 1, 2, 3]
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert not d.has_node_y
  assert not d.has_graph_x
  assert d.has_graph_y

  assert d.graph_y.dtype == np.int32
  assert np.array_equal(d.graph_y, np.array([0, 1, 2, 3]))


def test_ToNetworkx_node_and_edge_count(graph: nx.MultiDiGraph):
  g = graph_tuple.GraphTuple.CreateFromNetworkX(graph).ToNetworkx()

  assert g.number_of_nodes() == 5
  assert g.number_of_edges() == 4


def test_ToNetworkx_edge_flow(graph: nx.MultiDiGraph):
  g = graph_tuple.GraphTuple.CreateFromNetworkX(graph).ToNetworkx()

  assert g.edges[0, 1, programl_pb2.Edge.CALL]["flow"] == programl_pb2.Edge.CALL
  assert (
    g.edges[1, 2, programl_pb2.Edge.CONTROL]["flow"]
    == programl_pb2.Edge.CONTROL
  )
  assert (
    g.edges[2, 3, programl_pb2.Edge.CONTROL]["flow"]
    == programl_pb2.Edge.CONTROL
  )
  assert g.edges[4, 3, programl_pb2.Edge.DATA]["flow"] == programl_pb2.Edge.DATA


def test_ToNetworkx_edge_position(graph: nx.MultiDiGraph):
  g = graph_tuple.GraphTuple.CreateFromNetworkX(graph).ToNetworkx()

  assert g.edges[0, 1, programl_pb2.Edge.CALL]["position"] == 0
  assert g.edges[1, 2, programl_pb2.Edge.CONTROL]["position"] == 0
  assert g.edges[2, 3, programl_pb2.Edge.CONTROL]["position"] == 0
  assert g.edges[4, 3, programl_pb2.Edge.DATA]["position"] == 0


def test_ToNetworkx_node_x(graph: nx.MultiDiGraph):
  g = graph_tuple.GraphTuple.CreateFromNetworkX(graph).ToNetworkx()

  assert g.nodes[0]["discrete_x"] == [4, 0]
  assert g.nodes[1]["discrete_x"] == [0, 0]
  assert g.nodes[2]["discrete_x"] == [1, 0]
  assert g.nodes[3]["discrete_x"] == [2, 1]
  assert g.nodes[4]["discrete_x"] == [3, 0]


def test_ToNetworkx_node_y(graph: nx.MultiDiGraph):
  g = graph_tuple.GraphTuple.CreateFromNetworkX(graph).ToNetworkx()

  assert g.nodes[0]["real_y"] == []
  assert g.nodes[1]["real_y"] == []
  assert g.nodes[2]["real_y"] == []
  assert g.nodes[3]["real_y"] == []
  assert g.nodes[4]["real_y"] == []


def test_ToNetworkx_graph_x(graph: nx.MultiDiGraph):
  g = graph_tuple.GraphTuple.CreateFromNetworkX(graph).ToNetworkx()

  assert g.graph["discrete_x"] == []


def test_ToNetworkx_graph_y(graph: nx.MultiDiGraph):
  g = graph_tuple.GraphTuple.CreateFromNetworkX(graph).ToNetworkx()

  assert g.graph["discrete_y"] == []


def CreateRandomNetworkx() -> programl_pb2.ProgramGraph:
  """Generate a random graph proto."""
  g = random_cdfg_generator.FastCreateRandom()
  proto = networkx_to_protos.NetworkXGraphToProgramGraphProto(g)
  return programl.ProgramGraphToNetworkX(proto)


@decorators.loop_for(seconds=10)
def fuzz_graph_tuple_networkx():
  g = CreateRandomNetworkx()
  t = graph_tuple.GraphTuple.CreateFromNetworkX(g)
  g_out = t.ToNetworkx()
  assert g.number_of_nodes() == g_out.number_of_nodes()
  assert g.number_of_edges() == g_out.number_of_edges()


if __name__ == "__main__":
  test.Main()
