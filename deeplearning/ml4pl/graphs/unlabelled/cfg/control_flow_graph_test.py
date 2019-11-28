"""Unit tests for //deeplearning/ml4pl/graphs/unlabelled/cfg:control_flow_graph."""
import pytest

from deeplearning.ml4pl.graphs.unlabelled.cfg import control_flow_graph
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


def test_ToSuccessorsString_straight_line_graph():
  """Test successors list with a back edge in the graph."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #      A --> B --> C
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C", exit=True)
  g.add_edge(0, 1)
  g.add_edge(1, 2)
  assert (
    g.ToSuccessorsString()
    == """\
A: B C
B: C
C: """
  )


def test_ControlFlowGraph_ToSuccessorsString_if_else_loop():
  """Test successors of an if-else loop graph."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #     +----> B -----+
  #     |             |
  #     |             v
  #     A             D
  #     |             ^
  #     |             |
  #     +----> C -----+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C")
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(0, 2)
  g.add_edge(1, 3)
  g.add_edge(2, 3)
  assert (
    g.ToSuccessorsString()
    == """\
A: B C D
B: D
C: D
D: """
  )


def test_ControlFlowGraph_ToSuccessorsString_while_loop():
  """Test successors of a while loop graph."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #     +--------+
  #     |        |
  #     v        |
  #     A+------>B       C
  #     |                ^
  #     |                |
  #     +----------------+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C", exit=True)
  g.add_edge(0, 1)
  g.add_edge(1, 0)
  g.add_edge(0, 2)
  # TODO(cec): I don't beleive these. Why isn't self included?
  assert (
    g.ToSuccessorsString()
    == """\
A: B C
B: A C
C: """
  )


def test_ControlFlowGraph_ToSuccessorsString_while_loop_with_exit():
  """Test successors of a while loop with an if branch exit."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #     +----------------+
  #     |                |
  #     v                |
  #     A+------>B+----->C       D
  #     |        |               ^
  #     |        |               |
  #     +------->+---------------+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C")
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(1, 2)
  g.add_edge(2, 0)
  g.add_edge(0, 3)
  g.add_edge(1, 3)
  # TODO(cec): I don't beleive these. Why isn't self included?
  assert (
    g.ToSuccessorsString()
    == """\
A: B C D
B: A C D
C: A B D
D: """
  )


def test_ControlFlowGraph_ToSuccessorsString_irreducible_loop():
  """Test successors of an irreducible graph."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #              +-------+
  #              |       |
  #              v       |
  #     A------->B+----->C
  #     |        |       ^
  #     |        |       |
  #     |        v       |
  #     |        D       |
  #     |                |
  #     +----------------+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C")
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(0, 2)
  g.add_edge(1, 2)
  g.add_edge(2, 1)
  g.add_edge(1, 3)
  assert (
    g.ToSuccessorsString()
    == """\
A: B C D
B: C D
C: B D
D: """
  )


def test_ToNeighborsString_straight_line_graph():
  """Test neighbors list with a back edge in the graph."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #      A --> B --> C
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C", exit=True)
  g.add_edge(0, 1)
  g.add_edge(1, 2)
  assert (
    g.ToNeighborsString()
    == """\
A: B
B: C
C: """
  )


def test_ControlFlowGraph_ToNeighborsString_if_else_loop():
  """Test neighbors of an if-else loop graph."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #     +----> B -----+
  #     |             |
  #     |             v
  #     A             D
  #     |             ^
  #     |             |
  #     +----> C -----+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C")
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(0, 2)
  g.add_edge(1, 3)
  g.add_edge(2, 3)
  assert (
    g.ToNeighborsString()
    == """\
A: B C
B: D
C: D
D: """
  )


def test_ControlFlowGraph_ToNeighborsString_while_loop():
  """Test neighbors of a while loop graph."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #     +--------+
  #     |        |
  #     v        |
  #     A+------>B       C
  #     |                ^
  #     |                |
  #     +----------------+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C", exit=True)
  g.add_edge(0, 1)
  g.add_edge(1, 0)
  g.add_edge(0, 2)
  assert (
    g.ToNeighborsString()
    == """\
A: B C
B: A
C: """
  )


def test_ControlFlowGraph_ToNeighborsString_while_loop_with_exit():
  """Test neighbors of a while loop with an if branch exit."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #     +----------------+
  #     |                |
  #     v                |
  #     A+------>B+----->C       D
  #     |        |               ^
  #     |        |               |
  #     +------->+---------------+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C")
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(1, 2)
  g.add_edge(2, 0)
  g.add_edge(0, 3)
  g.add_edge(1, 3)
  assert (
    g.ToNeighborsString()
    == """\
A: B D
B: C D
C: A
D: """
  )


def test_ControlFlowGraph_ToNeighborsString_irreducible_loop():
  """Test neighbors of an irreducible graph."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #              +-------+
  #              |       |
  #              v       |
  #     A------->B+----->C
  #     |        |       ^
  #     |        |       |
  #     |        v       |
  #     |        D       |
  #     |                |
  #     +----------------+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C")
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(0, 2)
  g.add_edge(1, 2)
  g.add_edge(2, 1)
  g.add_edge(1, 3)
  assert (
    g.ToNeighborsString()
    == """\
A: B C
B: C D
C: B
D: """
  )


def test_ControlFlowGraph_validate_empty_graph():
  """Test that empty graph is invalid."""
  g = control_flow_graph.ControlFlowGraph()
  with pytest.raises(control_flow_graph.NotEnoughNodes) as e_ctx:
    g.ValidateControlFlowGraph()
  assert str(e_ctx.value) == "Function `cfg` has no nodes"
  assert not g.IsValidControlFlowGraph()


def test_ControlFlowGraph_validate_one_node():
  """Test that single-node graph is valid."""
  g = control_flow_graph.ControlFlowGraph()
  g.add_node(0, name="A", entry=True, exit=True)
  g.ValidateControlFlowGraph()


def test_ControlFlowGraph_IsValidControlFlowGraph_disconnected_graph():
  """A disconnected graph is not valid."""
  g = control_flow_graph.ControlFlowGraph()
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B", exit=True)
  g.add_node(2, name="C")
  g.add_edge(0, 1)
  with pytest.raises(control_flow_graph.UnconnectedNode) as e_ctx:
    g.ValidateControlFlowGraph()
  assert str(e_ctx.value) == "Unconnected node 'C'"
  assert not g.IsValidControlFlowGraph()


def test_ControlFlowGraph_IsValidControlFlowGraph_no_path_from_entry_to_exit():
  """A disconnected graph is not valid."""
  # Graph: A -> B <- C
  g = control_flow_graph.ControlFlowGraph()
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C", exit=True)
  g.add_edge(0, 1)
  g.add_edge(2, 1)
  with pytest.raises(
    control_flow_graph.MalformedControlFlowGraphError
  ) as e_ctx:
    g.ValidateControlFlowGraph()
  assert str(e_ctx.value) == (
    "No path from entry node 'A' to exit node 'C' " "in function `cfg`"
  )
  assert not g.IsValidControlFlowGraph()


def test_ControlFlowGraph_IsValidControlFlowGraph_invalid_degrees():
  """Test that a graph where two nodes could be fused is invalid."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #      A --> B --> C
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C", exit=True)
  g.add_edge(0, 1)
  g.add_edge(1, 2)
  with pytest.raises(control_flow_graph.InvalidNodeDegree) as e_ctx:
    g.ValidateControlFlowGraph()
  assert str(e_ctx.value) == "outdegree(A) = 1, indegree(B) = 1"
  assert not g.IsValidControlFlowGraph()


def test_ControlFlowGraph_IsValidControlFlowGraph_unamed_nodes():
  """Test that all nodes in a graph must have a name."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #     +----> B -----+
  #     |             |
  #     |             v
  #     A             D
  #     |             ^
  #     |             |
  #     +---->   -----+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2)
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(0, 2)
  g.add_edge(1, 3)
  g.add_edge(2, 3)
  with pytest.raises(control_flow_graph.MissingNodeName) as e_ctx:
    g.ValidateControlFlowGraph()
  assert str(e_ctx.value) == "Node 2 has no name in function `cfg`"
  assert not g.IsValidControlFlowGraph()


def test_ControlFlowGraph_IsValidControlFlowGraph_duplicate_names():
  """Test that duplicate names is an error."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #     +----> B -----+
  #     |             |
  #     |             v
  #     A             D
  #     |             ^
  #     |             |
  #     +----> B -----+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="B")
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(0, 2)
  g.add_edge(1, 3)
  g.add_edge(2, 3)
  with pytest.raises(control_flow_graph.DuplicateNodeName) as e_ctx:
    g.ValidateControlFlowGraph()
  assert str(e_ctx.value) == "Duplicate node name 'B' in function `cfg`"
  assert not g.IsValidControlFlowGraph()


def test_ControlFlowGraph_IsValidControlFlowGraph_exit_block_has_output():
  """Test that an if-else loop graph is valid."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #     +----> B -----+
  #     |             |
  #     |             v
  #     A<-----------+D
  #     |             ^
  #     |             |
  #     +----> C -----+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C")
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(0, 2)
  g.add_edge(1, 3)
  g.add_edge(2, 3)
  g.add_edge(3, 0)
  with pytest.raises(control_flow_graph.InvalidNodeDegree) as e_ctx:
    g.ValidateControlFlowGraph()
  assert str(e_ctx.value) == "Exit block outdegree(D) = 1 in function `cfg`"
  assert not g.IsValidControlFlowGraph()


def test_ControlFlowGraph_IsValidControlFlowGraph_if_else_loop():
  """Test that an if-else loop graph is valid."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #     +----> B -----+
  #     |             |
  #     |             v
  #     A             D
  #     |             ^
  #     |             |
  #     +----> C -----+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C")
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(0, 2)
  g.add_edge(1, 3)
  g.add_edge(2, 3)
  assert g.ValidateControlFlowGraph() == g
  assert g.IsValidControlFlowGraph()


def test_ControlFlowGraph_IsValidControlFlowGraph_while_loop():
  """Test that a while loop graph is valid."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #     +--------+
  #     |        |
  #     v        |
  #     A+------>B       C
  #     |                ^
  #     |                |
  #     +----------------+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C", exit=True)
  g.add_edge(0, 1)
  g.add_edge(1, 0)
  g.add_edge(0, 2)
  assert g.ValidateControlFlowGraph() == g
  assert g.IsValidControlFlowGraph()


def test_ControlFlowGraph_IsValidControlFlowGraph_while_loop_with_exit():
  """Test that a while loop with an if branch exit is valid."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #     +----------------+
  #     |                |
  #     v                |
  #     A+------>B+----->C       D
  #     |        |               ^
  #     |        |               |
  #     +------->+---------------+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C")
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(1, 2)
  g.add_edge(2, 0)
  g.add_edge(0, 3)
  g.add_edge(1, 3)
  assert g.ValidateControlFlowGraph() == g
  assert g.IsValidControlFlowGraph()


def test_ControlFlowGraph_IsValidControlFlowGraph_irreducible_loop():
  """Test that an irreducible graph is valid."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #              +-------+
  #              |       |
  #              v       |
  #     A------->B+----->C
  #     |        |       ^
  #     |        |       |
  #     |        v       |
  #     |        D       |
  #     |                |
  #     +----------------+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C")
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(0, 2)
  g.add_edge(1, 2)
  g.add_edge(2, 1)
  g.add_edge(1, 3)
  assert g.ValidateControlFlowGraph() == g
  assert g.IsValidControlFlowGraph()


def test_ControlFlowGraph_entry_block():
  """Test entry block."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #     +----> B -----+
  #     |             |
  #     |             v
  #     A             D
  #     |             ^
  #     |             |
  #     +----> C -----+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C")
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(0, 2)
  g.add_edge(1, 3)
  g.add_edge(2, 3)
  assert g.entry_block == 0


def test_ControlFlowGraph_exit_block():
  """Test exit block."""
  g = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #     +----> B -----+
  #     |             |
  #     |             v
  #     A             D
  #     |             ^
  #     |             |
  #     +----> C -----+
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C")
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(0, 2)
  g.add_edge(1, 3)
  g.add_edge(2, 3)
  assert g.exit_blocks == [3]


def test_ControlFlowGraph_equal():
  """Test that equal graphs can be compared."""
  # Graph 1: A --> B
  g1 = control_flow_graph.ControlFlowGraph()
  g1.add_node(0, name="A", entry=True)
  g1.add_node(1, name="B", exit=True)
  g1.add_edge(0, 1)

  # Graph 2: A --> B
  g2 = control_flow_graph.ControlFlowGraph()
  g2.add_node(0, name="A", entry=True)
  g2.add_node(1, name="B", exit=True)
  g2.add_edge(0, 1)

  assert g1 == g2


def test_ControlFlowGraph_unequal_nodes():
  """Test that graphs with unequal nodes are not equal."""
  # Graph 1: A --> B    C
  g1 = control_flow_graph.ControlFlowGraph()
  g1.add_node(0, name="A", entry=True)
  g1.add_node(1, name="B", exit=True)
  g1.add_node(2, name="C", exit=True)
  g1.add_edge(0, 1)

  # Graph 2: A --> B
  g2 = control_flow_graph.ControlFlowGraph()
  g2.add_node(0, name="A", entry=True)
  g2.add_node(1, name="B", exit=True)
  g2.add_edge(0, 1)

  assert g1 != g2


def test_ControlFlowGraph_unequal_edges():
  """Test that graphs with unequal edges are not equal."""
  # Graph 1: A --> B
  g1 = control_flow_graph.ControlFlowGraph()
  g1.add_node(0, name="A", entry=True)
  g1.add_node(1, name="B", exit=True)
  g1.add_edge(0, 1)

  # Graph 2: B --> A
  g2 = control_flow_graph.ControlFlowGraph()
  g2.add_node(0, name="A", entry=True)
  g2.add_node(1, name="B", exit=True)
  g2.add_edge(1, 0)

  assert g1 != g2


def test_ControlFlowGraph_unequal_graph_names_are_equal():
  """Test that graph names are not used in comparison."""
  # Graph 1: A --> B
  g1 = control_flow_graph.ControlFlowGraph(name="foo")
  g1.add_node(0, name="A", entry=True)
  g1.add_node(1, name="B", exit=True)
  g1.add_edge(0, 1)

  # Graph 2: A --> B
  g2 = control_flow_graph.ControlFlowGraph(name="bar")
  g2.add_node(0, name="A", entry=True)
  g2.add_node(1, name="B", exit=True)
  g2.add_edge(0, 1)

  assert g1 == g2


def test_ControlFlowGraph_unequal_edge_data():
  """Test that edge data is used in comparison."""
  # Graph 1: A --> B
  g1 = control_flow_graph.ControlFlowGraph(name="foo")
  g1.add_node(0, name="A", exit=True)
  g1.add_node(1, name="B")
  g1.add_edge(0, 1)

  # Graph 2: A --> B
  g2 = control_flow_graph.ControlFlowGraph(name="bar")
  g2.add_node(0, name="A")
  g2.add_node(1, name="B", entry=True)
  g2.add_edge(0, 1)

  assert g1 != g2


def test_ControlFlowGraph_ToProto_FromProto_equivalency():
  """Test that conversion to and from proto preserves values."""
  g1 = control_flow_graph.ControlFlowGraph()
  # Graph:
  #
  #     +----> B -----+
  #     |             |
  #     |             v
  #     A             D
  #     |             ^
  #     |             |
  #     +----> C -----+
  g1.add_node(0, name="A", entry=True)
  g1.add_node(1, name="B")
  g1.add_node(2, name="C")
  g1.add_node(3, name="D", exit=True)
  g1.add_edge(0, 1)
  g1.add_edge(0, 2)
  g1.add_edge(1, 3)
  g1.add_edge(2, 3)

  proto = g1.ToProto()

  g2 = control_flow_graph.ControlFlowGraph.FromProto(proto)

  assert g1 == g2

  # Graph names are not used in equality checks.
  assert g1.name == g2.name


def test_ControlFlowGraph_edge_density():
  """Test edge density property."""
  # Graph:
  #
  #     +----> B -----+
  #     |             |
  #     |             v
  #     A             D
  #     |             ^
  #     |             |
  #     +----> C -----+
  g = control_flow_graph.ControlFlowGraph()
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C")
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(0, 2)
  g.add_edge(1, 3)
  g.add_edge(2, 3)

  assert g.edge_density == pytest.approx(1 / 4)

  # Add A -> D path.
  g.add_edge(0, 3)
  assert g.edge_density == pytest.approx(5 / 16)


def test_ControlFlowGraph_undirected_diameter():
  """Test undirected_diameter property."""
  # Graph:
  #
  #     +----> B -----+
  #     |             |
  #     |             v
  #     A             D
  #     |             ^
  #     |             |
  #     +----> C -----+
  g = control_flow_graph.ControlFlowGraph()
  g.add_node(0, name="A", entry=True)
  g.add_node(1, name="B")
  g.add_node(2, name="C")
  g.add_node(3, name="D", exit=True)
  g.add_edge(0, 1)
  g.add_edge(0, 2)
  g.add_edge(1, 3)
  g.add_edge(2, 3)

  assert g.undirected_diameter == 2

  # Increase the diameter by adding a new E node and D -> E path.
  g.add_node(4, name="E")
  g.add_edge(3, 4)
  assert g.undirected_diameter == 3


def test_ControlFlowGraph_equivalent_hashes():
  """Test equivalent hashes, despite different graph names."""
  # Graph 1: A --> B
  g1 = control_flow_graph.ControlFlowGraph(name="foo")
  g1.add_node(0, name="A", entry=True)
  g1.add_node(1, name="B", exit=True)
  g1.add_edge(0, 1)

  # Graph 2: A --> B
  g2 = control_flow_graph.ControlFlowGraph(name="bar")
  g2.add_node(0, name="A", entry=True)
  g2.add_node(1, name="B", exit=True)
  g2.add_edge(0, 1)

  assert hash(g1) == hash(g2)


def test_ControlFlowGraph_node_name_changes_hash():
  """Test that hash depends on node name."""
  g1 = control_flow_graph.ControlFlowGraph()
  g1.add_node(0, name="A", entry=True)

  g2 = control_flow_graph.ControlFlowGraph()
  g2.add_node(0, name="B", entry=True)

  assert hash(g1) != hash(g2)


def test_ControlFlowGraph_node_attribute_changes_hash():
  """Test that hash depends on node attributes."""
  g1 = control_flow_graph.ControlFlowGraph()
  g1.add_node(0, name="A")

  g2 = g1.copy()
  assert hash(g1) == hash(g2)

  g2.nodes[0]["entry"] = True
  assert hash(g1) != hash(g2)


def test_ControlFlowGraph_IsomorphicHash_equivalency():
  """Test equivalent hashes, despite different attributes."""
  # Graph 1: A --> B
  g1 = control_flow_graph.ControlFlowGraph(name="foo")
  g1.add_node(0, name="A", entry=True)
  g1.add_node(1, name="B", exit=True)
  g1.add_edge(0, 1)

  # Graph 2: C --> D
  g2 = control_flow_graph.ControlFlowGraph(name="bar")
  g2.add_node(0, name="C", entry=True)
  g2.add_node(1, name="D", exit=True)
  g2.add_edge(0, 1)

  assert g1.IsomorphicHash() == g2.IsomorphicHash()


if __name__ == "__main__":
  test.Main()
