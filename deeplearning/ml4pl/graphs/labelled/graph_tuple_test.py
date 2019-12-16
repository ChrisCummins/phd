"""Unit tests for //deeplearning/ml4pl/graphs/labelled:graph_tuple."""
import pickle
import random
from typing import Tuple

import networkx as nx
import numpy as np

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled import graph_tuple
from deeplearning.ml4pl.testing import random_graph_tuple_generator
from deeplearning.ml4pl.testing import random_networkx_generator
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
  g.add_edge(4, 3, flow=programl_pb2.Edge.DATA, position=1)
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
  assert np.array_equal(d.edge_positions[programl_pb2.Edge.DATA], np.array([1]))
  assert np.array_equal(d.edge_positions[programl_pb2.Edge.CALL], np.array([0]))


def test_CreateFromNetworkX_edge_position_max(graph: nx.MultiDiGraph):
  """Test edge position max."""
  d = graph_tuple.GraphTuple.CreateFromNetworkX(graph)

  assert d.edge_position_max == 1


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

  assert d.node_x.dtype == np.int64
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
  assert d.node_y.dtype == np.int64
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

  assert d.graph_x.dtype == np.int64
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

  assert d.graph_y.dtype == np.int64
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
  assert g.edges[4, 3, programl_pb2.Edge.DATA]["position"] == 1


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


@test.Fixture(
  scope="function", params=list(random_networkx_generator.EnumerateTestSet()),
)
def real_nx_graph(request) -> nx.MultiDiGraph:
  return request.param


def test_on_real_graph(real_nx_graph: nx.MultiDiGraph):
  """Test nx -> graph_tuple -> nx conversion on real graphs."""
  # Create graph tuple from networkx.
  t = graph_tuple.GraphTuple.CreateFromNetworkX(real_nx_graph)
  try:
    assert t.node_count == real_nx_graph.number_of_nodes()
    assert t.edge_count == real_nx_graph.number_of_edges()
  except AssertionError:
    fs.Write("/tmp/graph_in.pickle", pickle.dumps(real_nx_graph))
    fs.Write("/tmp/graph_tuple_out.pickle", pickle.dumps(t))
    raise

  # Convert graph tuple back to networkx.
  g = t.ToNetworkx()
  try:
    assert g.number_of_nodes() == real_nx_graph.number_of_nodes()
    # TODO(github.com/ChrisCummins/ProGraML/issues/36): Fix me.
    # assert g.number_of_edges() == real_nx_graph.number_of_edges()
  except AssertionError:
    fs.Write("/tmp/graph_in.pickle", pickle.dumps(real_nx_graph))
    fs.Write("/tmp/graph_out.pickle", pickle.dumps(g))
    raise


# Disjoint graph tests:


@decorators.loop_for(seconds=3)
@test.Parametrize(
  "node_x_dimensionality", (1, 3), namer=lambda x: f"node_x_dimensionality:{x}"
)
@test.Parametrize(
  "node_y_dimensionality", (0, 3), namer=lambda x: f"node_y_dimensionality:{x}"
)
@test.Parametrize(
  "graph_x_dimensionality",
  (0, 3),
  namer=lambda x: f"graph_x_dimensionality:{x}",
)
@test.Parametrize(
  "graph_y_dimensionality",
  (0, 3),
  namer=lambda x: f"graph_y_dimensionality:{x}",
)
def test_FromGraphTuples_single_tuple(
  node_x_dimensionality: int,
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
) -> graph_tuple.GraphTuple:
  """Test disjoint graph creation from a single graph."""
  t = random_graph_tuple_generator.CreateRandomGraphTuple(
    node_x_dimensionality=node_x_dimensionality,
    node_y_dimensionality=node_y_dimensionality,
    graph_x_dimensionality=graph_x_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
  )

  # Create a disjoint GraphTuple.
  d = graph_tuple.GraphTuple.FromGraphTuples([t])

  try:
    assert d.disjoint_graph_count == 1
    assert d.node_count == t.node_count
    assert d.edge_count == t.edge_count
    assert d.edge_position_max == t.edge_position_max

    # Only a single graph means an array of all zeros.
    assert np.array_equal(
      d.disjoint_nodes_list, np.zeros(d.node_count, dtype=np.int32)
    )

    # Dimensionalities.
    assert d.node_x_dimensionality == t.node_x_dimensionality
    assert d.node_y_dimensionality == t.node_y_dimensionality
    assert d.graph_x_dimensionality == t.graph_x_dimensionality
    assert d.graph_y_dimensionality == t.graph_y_dimensionality

    # Feature and label vectors.
    assert np.array_equal(d.node_x, t.node_x)
    assert np.array_equal(d.node_y, t.node_y)
    if t.has_graph_x:
      assert np.array_equal(d.graph_x[0], t.graph_x)
    else:
      assert d.graph_x == t.graph_x
    if t.has_graph_y:
      assert np.array_equal(d.graph_y[0], t.graph_y)
    else:
      assert d.graph_y == t.graph_y
  except AssertionError:
    fs.Write("/tmp/graph_tuple_in.pickle", pickle.dumps(t))
    fs.Write("/tmp/graph_tuple_out.pickle", pickle.dumps(d))
    raise


@decorators.loop_for(seconds=3)
@test.Parametrize(
  "node_x_dimensionality", (1, 3), namer=lambda x: f"node_x_dimensionality:{x}"
)
@test.Parametrize(
  "node_y_dimensionality", (0, 3), namer=lambda x: f"node_y_dimensionality:{x}"
)
@test.Parametrize(
  "graph_x_dimensionality",
  (0, 3),
  namer=lambda x: f"graph_x_dimensionality:{x}",
)
@test.Parametrize(
  "graph_y_dimensionality",
  (0, 3),
  namer=lambda x: f"graph_y_dimensionality:{x}",
)
def test_FromGraphTuples_two_tuples(
  node_x_dimensionality: int,
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
):
  """Test disjoint graph creation from a pair of input graphs."""
  tuples_in = [
    random_graph_tuple_generator.CreateRandomGraphTuple(
      node_x_dimensionality=node_x_dimensionality,
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=graph_x_dimensionality,
      graph_y_dimensionality=graph_y_dimensionality,
    ),
    random_graph_tuple_generator.CreateRandomGraphTuple(
      node_x_dimensionality=node_x_dimensionality,
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
    assert disjoint_tuple.edge_position_max == max(
      t.edge_position_max for t in tuples_in
    )

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
  except AssertionError:
    fs.Write("/tmp/graph_tuples_in.pickle", pickle.dumps(tuples_in))
    fs.Write("/tmp/graph_tuple_out.pickle", pickle.dumps(disjoint_tuple))
    app.Error(
      "Assertion failed! Wrote graphs to /tmp/graph_tuples_in.pickle "
      "and /tmp/graph_tuple_out.pickle"
    )
    raise


@decorators.loop_for(seconds=3)
@test.Parametrize("dimensionalities", ((0, 2), (2, 0)))
@test.Parametrize("copy", (False, True))
def test_SetFeaturesAndLabels(dimensionalities: Tuple[int, int], copy: bool):
  """Test new label setting."""
  node_y_dimensionality, graph_y_dimensionality = dimensionalities
  in_tuple = random_graph_tuple_generator.CreateRandomGraphTuple(
    node_y_dimensionality=node_y_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
  )
  old_node_y = np.copy(in_tuple.node_y)
  old_graph_y = np.copy(in_tuple.graph_y)

  new_node_y = np.random.rand(
    in_tuple.node_count, in_tuple.node_y_dimensionality
  )
  new_graph_y = np.random.rand(in_tuple.graph_y_dimensionality)
  out_tuple = in_tuple.SetFeaturesAndLabels(
    node_y=np.copy(new_node_y), graph_y=np.copy(new_graph_y), copy=copy,
  )

  # Test that input tuple is not modified.
  assert np.array_equal(in_tuple.node_y, old_node_y)
  assert np.array_equal(in_tuple.graph_y, old_graph_y)

  # Test that output tuple has correct labels.
  assert np.array_equal(out_tuple.node_y, new_node_y)
  assert np.array_equal(out_tuple.graph_y, new_graph_y)


# Fuzzers:


@decorators.loop_for(seconds=30)
def test_fuzz_graph_tuple_networkx():
  """Fuzz graph tuples with randomly generated graphs."""
  node_x_dimensionality = random.randint(1, 3)
  node_y_dimensionality = random.randint(0, 3)
  graph_x_dimensionality = random.randint(0, 3)
  graph_y_dimensionality = random.randint(0, 3)
  graph_in = random_networkx_generator.CreateRandomGraph(
    node_x_dimensionality=node_x_dimensionality,
    node_y_dimensionality=node_y_dimensionality,
    graph_x_dimensionality=graph_x_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
  )
  t = graph_tuple.GraphTuple.CreateFromNetworkX(graph_in)
  try:
    assert t.node_count == graph_in.number_of_nodes()
    assert t.edge_count == graph_in.number_of_edges()
    assert t.node_x_dimensionality == node_x_dimensionality
    assert t.node_y_dimensionality == node_y_dimensionality
    assert t.graph_x_dimensionality == graph_x_dimensionality
    assert t.graph_y_dimensionality == graph_y_dimensionality
    assert t.edge_position_max == max(
      position for _, _, position in graph_in.edges(data="position")
    )
  except AssertionError:
    fs.Write("/tmp/graph_in.pickle", pickle.dumps(graph_in))
    fs.Write("/tmp/graph_tuple_out.pickle", pickle.dumps(t))
    app.Error(
      "Assertion failed! Wrote graphs to /tmp/graphs_in.pickle "
      "and /tmp/graph_tuple_out.pickle"
    )
    raise

  g_out = t.ToNetworkx()
  try:
    assert graph_in.number_of_nodes() == g_out.number_of_nodes()
    assert graph_in.number_of_edges() == g_out.number_of_edges()
    assert len(g_out.nodes[0]["x"]) == node_x_dimensionality
    assert len(g_out.nodes[0]["y"]) == node_y_dimensionality
    assert len(g_out.graph["x"]) == graph_x_dimensionality
    assert len(g_out.graph["y"]) == graph_y_dimensionality
  except AssertionError:
    fs.Write("/tmp/graph_in.pickle", pickle.dumps(graph_in))
    fs.Write("/tmp/graph_out.pickle", pickle.dumps(g_out))
    app.Error(
      "Assertion failed! Wrote graphs to /tmp/graphs_in.pickle "
      "and /tmp/graphs_out.pickle"
    )
    raise


@decorators.loop_for(seconds=30)
def test_fuzz_disjoint_graph_tuples():
  """Fuzz graph tuples with randomly generated graphs."""
  disjoint_graph_count = random.randint(2, 10)
  node_x_dimensionality = random.randint(1, 3)
  node_y_dimensionality = random.randint(0, 3)
  graph_x_dimensionality = random.randint(0, 3)
  graph_y_dimensionality = random.randint(0, 3)

  graph_tuples_in = [
    random_graph_tuple_generator.CreateRandomGraphTuple(
      node_x_dimensionality=node_x_dimensionality,
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=graph_x_dimensionality,
      graph_y_dimensionality=graph_y_dimensionality,
    )
    for _ in range(disjoint_graph_count)
  ]

  t = graph_tuple.GraphTuple.FromGraphTuples(graph_tuples_in)

  try:
    assert t.disjoint_graph_count == disjoint_graph_count
    assert t.node_count == sum([d.node_count for d in graph_tuples_in])
    assert t.edge_count == sum([d.edge_count for d in graph_tuples_in])
    assert t.edge_position_max == max(
      d.edge_position_max for d in graph_tuples_in
    )
    assert t.node_x_dimensionality == node_x_dimensionality
    assert t.node_y_dimensionality == node_y_dimensionality
    assert t.graph_x_dimensionality == graph_x_dimensionality
    assert t.graph_y_dimensionality == graph_y_dimensionality
  except AssertionError:
    fs.Write("/tmp/graph_tuples_in.pickle", pickle.dumps(graph_tuples_in))
    fs.Write("/tmp/graph_tuple_out.pickle", pickle.dumps(t))
    raise

  graph_tuples_out = list(t.ToGraphTuples())
  try:
    assert len(graph_tuples_in) == len(graph_tuples_out)
    for tuple_in, tuple_out in zip(graph_tuples_in, graph_tuples_out):
      assert tuple_out.node_x_dimensionality == node_x_dimensionality
      assert tuple_out.node_y_dimensionality == node_y_dimensionality
      assert tuple_out.graph_x_dimensionality == graph_x_dimensionality
      assert tuple_out.graph_y_dimensionality == graph_y_dimensionality
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
  except AssertionError:
    fs.Write("/tmp/graph_tuples_in.pickle", pickle.dumps(graph_tuples_in))
    fs.Write("/tmp/graph_tuples_out.pickle", pickle.dumps(graph_tuples_out))
    raise


if __name__ == "__main__":
  test.Main()
