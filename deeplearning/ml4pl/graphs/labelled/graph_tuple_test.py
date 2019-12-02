"""Unit tests for //deeplearning/ml4pl/graphs/labelled:graph_tuple."""
import pickle
import random

import networkx as nx
import numpy as np

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled import graph_tuple
from deeplearning.ml4pl.graphs.migrate import networkx_to_protos
from deeplearning.ml4pl.graphs.unlabelled.cdfg import random_cdfg_generator
from labm8.py import app
from labm8.py import decorators
from labm8.py import fs
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def graph() -> nx.MultiDiGraph:
  g = nx.MultiDiGraph()
  g.add_node(0, type=programl_pb2.Node.STATEMENT, x=[4, 0])
  g.add_node(1, type=programl_pb2.Node.STATEMENT, x=[0, 0])
  g.add_node(2, type=programl_pb2.Node.STATEMENT, x=[1, 0])
  g.add_node(3, type=programl_pb2.Node.STATEMENT, x=[2, 1])
  g.add_node(4, type=programl_pb2.Node.IDENTIFIER, x=[3, 0])
  g.add_edge(0, 1, flow=programl_pb2.Edge.CALL, position=0)
  g.add_edge(1, 2, flow=programl_pb2.Edge.CONTROL, position=0)
  g.add_edge(2, 3, flow=programl_pb2.Edge.CONTROL, position=0)
  g.add_edge(4, 3, flow=programl_pb2.Edge.DATA, position=0)
  g.graph["x"] = []
  g.graph["y"] = []
  return g


# nx.MultiDiGraph -> GraphTuple.


def test_CreateFromNetworkX_adjacencies(graph: nx.MultiDiGraph):
  """Test adjacency list size and values."""
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert d.adjacencies.shape == (3,)

  assert d.adjacencies[programl_pb2.Edge.CONTROL].dtype == np.int32
  assert d.adjacencies[programl_pb2.Edge.DATA].dtype == np.int32
  assert d.adjacencies[programl_pb2.Edge.CALL].dtype == np.int32

  assert np.array_equal(
    d.adjacencies[programl_pb2.Edge.CONTROL], np.array([(1, 2), (2, 3)])
  )
  assert np.array_equal(
    d.adjacencies[programl_pb2.Edge.DATA], np.array([(4, 3)])
  )
  assert np.array_equal(
    d.adjacencies[programl_pb2.Edge.CALL], np.array([(0, 1)])
  )


def test_CreateFromNetworkX_edge_positions(graph: nx.MultiDiGraph):
  """Test edge positions size and values."""
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
  """Test node features dimensionality and values."""
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert not d.has_node_y
  assert not d.has_graph_x
  assert not d.has_graph_y

  assert d.node_x_dimensionality == 2
  assert d.node_y_dimensionality == 0
  assert d.graph_x_dimensionality == 0
  assert d.graph_y_dimensionality == 0

  assert d.node_x.dtype == np.int32
  assert np.array_equal(
    d.node_x, np.array([(4, 0), (0, 0), (1, 0), (2, 1), (3, 0),])
  )


def test_CreateFromNetworkX_node_y(graph: nx.MultiDiGraph):
  """Test node labels dimensionality and values."""

  graph.nodes[0]["y"] = [4, 1, 0]
  graph.nodes[1]["y"] = [3, 1, 0]
  graph.nodes[2]["y"] = [2, 1, 0]
  graph.nodes[3]["y"] = [1, 1, 0]
  graph.nodes[4]["y"] = [0, 1, 0]
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert d.has_node_y
  assert not d.has_graph_x
  assert not d.has_graph_y

  assert d.node_x_dimensionality == 2
  assert d.node_y_dimensionality == 3
  assert d.graph_x_dimensionality == 0
  assert d.graph_y_dimensionality == 0

  assert d.node_y.shape == (5, 3)
  assert d.node_y.dtype == np.int32
  assert np.array_equal(
    d.node_y, np.array([(4, 1, 0), (3, 1, 0), (2, 1, 0), (1, 1, 0), (0, 1, 0),])
  )


def test_CreateFromNetworkX_graph_x(graph: nx.MultiDiGraph):
  """Test graph features dimensionality and values."""
  graph.graph["x"] = [0, 1, 2, 3]
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert d.node_x_dimensionality == 2
  assert d.node_y_dimensionality == 0
  assert d.graph_x_dimensionality == 4
  assert d.graph_y_dimensionality == 0

  assert not d.has_node_y
  assert d.has_graph_x
  assert not d.has_graph_y

  assert d.graph_x.dtype == np.int32
  assert np.array_equal(d.graph_x, np.array([0, 1, 2, 3]))


def test_CreateFromNetworkX_graph_y(graph: nx.MultiDiGraph):
  """Test graph labels dimensionality and values."""

  graph.graph["y"] = [0, 1, 2, 3]
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert d.node_x_dimensionality == 2
  assert d.node_y_dimensionality == 0
  assert d.graph_x_dimensionality == 0
  assert d.graph_y_dimensionality == 4

  assert not d.has_node_y
  assert not d.has_graph_x
  assert d.has_graph_y

  assert d.graph_y.dtype == np.int32
  assert np.array_equal(d.graph_y, np.array([0, 1, 2, 3]))


def test_CreateFromNetworkX_node_count(graph: nx.MultiDiGraph):
  """Test the node count."""
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert d.node_count == 5


def test_CreateFromNetworkX_edge_counts(graph: nx.MultiDiGraph):
  """Test the typed edge counts."""
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert d.edge_count == 4
  assert d.control_edge_count == 2
  assert d.data_edge_count == 1
  assert d.call_edge_count == 1


def test_CreateFromNetworkX_not_disjoint(graph: nx.MultiDiGraph):
  """Test that a single graph is not disjoint."""
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert not d.is_disjoint_graph
  assert d.disjoint_graph_count == 1


# GraphTuple -> nx.MultiDiGraph tests.


def test_ToNetworkx_node_and_edge_count(graph: nx.MultiDiGraph):
  """Test graph shape."""

  g = graph_tuple.GraphTuple.CreateFromNetworkX(graph).ToNetworkx()

  assert g.number_of_nodes() == 5
  assert g.number_of_edges() == 4


def test_ToNetworkx_edge_flow(graph: nx.MultiDiGraph):
  """Test graph edge flows."""
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
  """Test graph edge positions."""
  g = graph_tuple.GraphTuple.CreateFromNetworkX(graph).ToNetworkx()

  assert g.edges[0, 1, programl_pb2.Edge.CALL]["position"] == 0
  assert g.edges[1, 2, programl_pb2.Edge.CONTROL]["position"] == 0
  assert g.edges[2, 3, programl_pb2.Edge.CONTROL]["position"] == 0
  assert g.edges[4, 3, programl_pb2.Edge.DATA]["position"] == 0


def test_ToNetworkx_node_x(graph: nx.MultiDiGraph):
  """Test graph node features."""
  g = graph_tuple.GraphTuple.CreateFromNetworkX(graph).ToNetworkx()

  assert g.nodes[0]["x"] == [4, 0]
  assert g.nodes[1]["x"] == [0, 0]
  assert g.nodes[2]["x"] == [1, 0]
  assert g.nodes[3]["x"] == [2, 1]
  assert g.nodes[4]["x"] == [3, 0]


def test_ToNetworkx_node_y(graph: nx.MultiDiGraph):
  """Test graph node labels."""
  g = graph_tuple.GraphTuple.CreateFromNetworkX(graph).ToNetworkx()

  assert g.nodes[0]["y"] == []
  assert g.nodes[1]["y"] == []
  assert g.nodes[2]["y"] == []
  assert g.nodes[3]["y"] == []
  assert g.nodes[4]["y"] == []


def test_ToNetworkx_graph_x_not_set(graph: nx.MultiDiGraph):
  """Test missing graph features."""
  g = graph_tuple.GraphTuple.CreateFromNetworkX(graph).ToNetworkx()

  assert g.graph["x"] == []


def test_ToNetworkx_graph_x_set(graph: nx.MultiDiGraph):
  """Test graph features."""
  graph.graph["x"] = [1, 2, 3]
  g = graph_tuple.GraphTuple.CreateFromNetworkX(graph).ToNetworkx()

  assert g.graph["x"] == [1, 2, 3]


def test_ToNetworkx_graph_y_not_set(graph: nx.MultiDiGraph):
  """Test missing graph labels."""
  g = graph_tuple.GraphTuple.CreateFromNetworkX(graph).ToNetworkx()

  assert g.graph["y"] == []


def test_ToNetworkx_graph_y_set(graph: nx.MultiDiGraph):
  """Test graph labels."""
  graph.graph["y"] = [1, 2]
  g = graph_tuple.GraphTuple.CreateFromNetworkX(graph).ToNetworkx()

  assert g.graph["y"] == [1, 2]


# Disjoint graph tests:


def CreateRandomGraphTuple(
  node_y_dimensionality: int = 0,
  graph_x_dimensionality: int = 0,
  graph_y_dimensionality: int = 0,
) -> graph_tuple.GraphTuple:
  """Generate a random graph tuple.

  Args:
    node_y_dimensionality: The dimensionality of node y vectors.
    graph_x_dimensionality: The dimensionality of graph x vectors.
    graph_y_dimensionality: The dimensionality of graph y vectors.

  Returns:
    A graph tuple.
  """

  def RandomList(n: int):
    return [random.randint(0, 10) for _ in range(n)]

  g = random_cdfg_generator.FastCreateRandom()
  proto = networkx_to_protos.NetworkXGraphToProgramGraphProto(g)

  if node_y_dimensionality:
    for node in proto.node:
      node.y[:] = RandomList(node_y_dimensionality)

  if graph_x_dimensionality:
    proto.x[:] = RandomList(graph_x_dimensionality)

  if graph_y_dimensionality:
    proto.y[:] = RandomList(graph_y_dimensionality)

  g = programl.ProgramGraphToNetworkX(proto)
  return graph_tuple.GraphTuple.CreateFromNetworkX(g)


@decorators.loop_for(seconds=5)
@test.Parametrize("node_y_dimensionality", (0, 1, 2, 3))
@test.Parametrize("graph_x_dimensionality", (0, 1, 2, 3))
@test.Parametrize("graph_y_dimensionality", (0, 1, 2, 3))
def test_FromGraphTuples_single_tuple(
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
) -> graph_tuple.GraphTuple:
  t = CreateRandomGraphTuple(
    node_y_dimensionality=node_y_dimensionality,
    graph_x_dimensionality=graph_x_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
  )
  # Sanity check the generated tuple.
  assert t.node_y_dimensionality == node_y_dimensionality
  assert t.graph_x_dimensionality == graph_x_dimensionality
  assert t.graph_y_dimensionality == graph_y_dimensionality

  # Create a disjoint GraphTuple.
  d = graph_tuple.GraphTuple.FromGraphTuples([t])

  try:
    assert d.disjoint_graph_count == 1
    assert d.node_count == t.node_count
    assert d.edge_count == t.edge_count

    # Only a single graph means an array of all zeros.
    assert np.array_equal(
      d.disjoint_nodes_list, np.zeros(d.node_count, dtype=np.int32)
    )

    # Dimensionalities.
    assert d.node_x_dimensionality == t.node_x_dimensionality
    assert d.node_y_dimensionality == t.node_y_dimensionality
    # assert d.graph_x_dimensionality == t.graph_x_dimensionality
    # assert d.graph_y_dimensionality == t.graph_y_dimensionality

    # Feature and label vectors.
    assert np.array_equal(d.node_x, t.node_x)
    assert np.array_equal(d.node_y, t.node_y)
    # assert np.array_equal(d.graph_x, t.graph_x)
    # assert np.array_equal(d.graph_y, t.graph_y)
  except AssertionError as e:
    fs.Write("/tmp/graph_tuple_in.pickle", pickle.dumps(t))
    fs.Write("/tmp/graph_tuple_out.pickle", pickle.dumps(d))
    app.Error(
      "Assertion failed! Wrote graphs to /tmp/graph_tuple_in.pickle "
      "and /tmp/graph_tuple_out.pickle"
    )
    raise e


@decorators.loop_for(seconds=5)
@test.Parametrize("node_y_dimensionality", (0, 1, 2, 3))
@test.Parametrize("graph_x_dimensionality", (0, 1, 2, 3))
@test.Parametrize("graph_y_dimensionality", (0, 1, 2, 3))
def test_FromGraphTuples_two_tuples(
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
) -> graph_tuple.GraphTuple:
  tuples_in = [
    CreateRandomGraphTuple(
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=graph_x_dimensionality,
      graph_y_dimensionality=graph_y_dimensionality,
    ),
    CreateRandomGraphTuple(
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=graph_x_dimensionality,
      graph_y_dimensionality=graph_y_dimensionality,
    ),
  ]

  # Create a disjoint GraphTuple.
  disjoint_tuple = graph_tuple.GraphTuple.FromGraphTuples(tuples_in)

  try:
    assert disjoint_tuple.disjoint_graph_count == 2
    assert disjoint_tuple.node_count == sum(t.node_count for t in tuples_in)
    assert disjoint_tuple.edge_count == sum(t.edge_count for t in tuples_in)

    # Only a single graph means an array of all zeros.
    assert np.array_equal(
      disjoint_tuple.disjoint_nodes_list,
      np.concatenate(
        [
          np.zeros(tuples_in[0].node_count, dtype=np.int32),
          np.ones(tuples_in[1].node_count, dtype=np.int32),
        ]
      ),
    )

    # Dimensionalities.
    assert (
      disjoint_tuple.node_x_dimensionality == tuples_in[0].node_x_dimensionality
    )
    assert disjoint_tuple.node_y_dimensionality == node_y_dimensionality
    assert disjoint_tuple.graph_x_dimensionality == graph_x_dimensionality
    assert disjoint_tuple.graph_y_dimensionality == graph_y_dimensionality
  except AssertionError as e:
    fs.Write("/tmp/graph_tuples_in.pickle", pickle.dumps(tuples_in))
    fs.Write("/tmp/graph_tuple_out.pickle", pickle.dumps(disjoint_tuple))
    app.Error(
      "Assertion failed! Wrote graphs to /tmp/graph_tuples_in.pickle "
      "and /tmp/graph_tuple_out.pickle"
    )
    raise e


# Fuzzers:


def CreateRandomNetworkx() -> programl_pb2.ProgramGraph:
  """Generate a random graph proto."""
  g = random_cdfg_generator.FastCreateRandom()
  proto = networkx_to_protos.NetworkXGraphToProgramGraphProto(g)
  return programl.ProgramGraphToNetworkX(proto)


@decorators.loop_for(seconds=60)
def test_fuzz_graph_tuple_networkx():
  """Fuzz graph tuples with randomly generated graphs."""
  graph_in = CreateRandomNetworkx()
  t = graph_tuple.GraphTuple.CreateFromNetworkX(graph_in)
  try:
    assert t.node_count == graph_in.number_of_nodes()
    assert t.edge_count == graph_in.number_of_edges()
  except AssertionError as e:
    fs.Write("/tmp/graph_in.pickle", pickle.dumps(graph_in))
    fs.Write("/tmp/graph_tuple_out.pickle", pickle.dumps(t))
    app.Error(
      "Assertion failed! Wrote graphs to /tmp/graphs_in.pickle "
      "and /tmp/graph_tuple_out.pickle"
    )
    raise e

  try:
    g_out = t.ToNetworkx()
    assert graph_in.number_of_nodes() == g_out.number_of_nodes()
    assert graph_in.number_of_edges() == g_out.number_of_edges()
  except AssertionError as e:
    fs.Write("/tmp/graph_in.pickle", pickle.dumps(graph_in))
    fs.Write("/tmp/graph_out.pickle", pickle.dumps(g_out))
    app.Error(
      "Assertion failed! Wrote graphs to /tmp/graphs_in.pickle "
      "and /tmp/graphs_out.pickle"
    )
    raise e


@decorators.loop_for(seconds=60)
def test_fuzz_disjoint_graph_tuples():
  """Fuzz graph tuples with randomly generated graphs."""
  num_graphs = random.randint(2, 30)
  graph_tuples_in = [CreateRandomGraphTuple() for _ in range(num_graphs)]

  t = graph_tuple.GraphTuple.FromGraphTuples(graph_tuples_in)

  try:
    assert t.disjoint_graph_count == num_graphs
    assert t.node_count == sum([d.node_count for d in graph_tuples_in])
    assert t.edge_count == sum([d.edge_count for d in graph_tuples_in])
    assert t.edge_position_max == max(
      [d.edge_position_max for d in graph_tuples_in]
    )
  except AssertionError as e:
    fs.Write("/tmp/graph_tuples_in.pickle", pickle.dumps(graph_tuples_in))
    fs.Write("/tmp/graph_tuple_out.pickle", pickle.dumps(t))
    app.Error(
      "Fuzzing failed! Wrote graphs to /tmp/graphs_tuples_in.pickle "
      "and /tmp/graph_tuple_out.pickle"
    )
    raise e

  graph_tuples_out = list(t.ToGraphTuples())
  try:
    assert len(graph_tuples_in) == len(graph_tuples_out)
    for tuple_in, tuple_out in zip(graph_tuples_in, graph_tuples_out):
      for i, (a, b) in enumerate(
        zip(tuple_in.adjacencies, tuple_out.adjacencies)
      ):
        assert a.shape == b.shape
        assert np.array_equal(a, b)
      for a, b in zip(tuple_in.edge_positions, tuple_out.edge_positions):
        assert a.shape == b.shape
        assert np.array_equal(a, b)
      assert np.array_equal(tuple_in.node_x, tuple_out.node_x)
      assert np.array_equal(tuple_in.node_y, tuple_out.node_y)
      assert np.array_equal(tuple_in.graph_x, tuple_out.graph_x)
      assert np.array_equal(tuple_in.graph_y, tuple_out.graph_y)
  except AssertionError as e:
    fs.Write("/tmp/graph_tuples_in.pickle", pickle.dumps(graph_tuples_in))
    fs.Write("/tmp/graph_tuples_out.pickle", pickle.dumps(graph_tuples_out))
    app.Error(
      "Fuzzing failed! Wrote graphs to /tmp/graphs_tuples_in.pickle "
      "and /tmp/graph_tuples_out.pickle"
    )
    raise e


if __name__ == "__main__":
  test.Main()
