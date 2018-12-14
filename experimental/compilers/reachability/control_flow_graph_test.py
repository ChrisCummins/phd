"""Unit tests for //experimental/compilers/reachability:control_flow_graph."""
import sys
import typing

import networkx as nx
import pytest
from absl import app
from absl import flags

from experimental.compilers.reachability import control_flow_graph


FLAGS = flags.FLAGS


def test_ControlFlowGraph_IsReachable_reachable():
  """Test reachable node."""
  g = control_flow_graph.ControlFlowGraph()
  g.add_edge(0, 1)
  assert g.IsReachable(0, 1)


def test_ControlFlowGraph_IsReachable_indirectly_reachable():
  """Test indirectly reachable node."""
  g = control_flow_graph.ControlFlowGraph()
  g.add_edge(0, 1)
  g.add_edge(1, 2)
  assert g.IsReachable(0, 2)


def test_ControlFlowGraph_IsReachable_unreachable():
  """Test unreachable node."""
  g = control_flow_graph.ControlFlowGraph()
  g.add_edge(0, 1)
  assert not g.IsReachable(1, 0)


def test_ControlFlowGraph_IsReachable_non_existent_node_raises_error():
  """Test that error is raised if node is not in graph."""
  g = control_flow_graph.ControlFlowGraph()
  with pytest.raises(nx.exception.NetworkXError):
    g.IsReachable(1, 0)


def test_ControlFlowGraph_Reachables_empty_graph():
  """An empty graph has no reachables."""
  g = control_flow_graph.ControlFlowGraph()
  assert list(g.Reachables(0)) == []


def test_ControlFlowGraph_Reachables_self_loop():
  """A self loop makes a node reachable."""
  g = control_flow_graph.ControlFlowGraph()
  g.add_edge(0, 0)
  assert list(g.Reachables(0)) == [True]


def test_ControlFlowGraph_Reachables_simple_graph():
  """An empty graph has no reachables."""
  g = control_flow_graph.ControlFlowGraph()
  g.add_edge(0, 1)
  g.add_edge(1, 2)
  assert list(g.Reachables(0)) == [False, True, True]
  assert list(g.Reachables(1)) == [False, False, True]
  assert list(g.Reachables(2)) == [False, False, False]


def test_ControlFlowGraph_Reachables_back_edge():
  """Test reachability with a back edge in the graph."""
  g = control_flow_graph.ControlFlowGraph()
  g.add_edge(0, 1)
  g.add_edge(1, 0)
  g.add_edge(1, 2)
  assert list(g.Reachables(0)) == [False, True, True]  # FIXME
  assert list(g.Reachables(1)) == [True, False, True]  # FIXME
  assert list(g.Reachables(2)) == [False, False, False]


def test_ControlFlowGraph_IsValidControlFlowGraph_empty_graph():
  """Test that empty graph is invalid."""
  g = control_flow_graph.ControlFlowGraph()
  assert not g.IsValidControlFlowGraph()


def test_ControlFlowGraph_IsValidControlFlowGraph_disconnected_graph():
  """Test simple_."""
  g = control_flow_graph.ControlFlowGraph()
  g.add_node(0, name='A')
  g.add_node(1, name='B')
  # FIXME: assert not g.IsValidControlFlowGraph()


def test_ControlFlowGraph_IsValidControlFlowGraph_unamed_nodes():
  """Test simple_."""
  g = control_flow_graph.ControlFlowGraph()
  g.add_edge(0, 1)
  assert not g.IsValidControlFlowGraph()


def test_ControlFlowGraph_IsValidControlFlowGraph_duplicate_names():
  """Test simple_."""
  g = control_flow_graph.ControlFlowGraph()
  g.add_node(0, name='A')
  g.add_node(1, name='A')
  g.add_edge(0, 1)
  assert not g.IsValidControlFlowGraph()


def test_ControlFlowGraph_IsValidControlFlowGraph_valid_graph():
  """Test that a simple graph is valid."""
  g = control_flow_graph.ControlFlowGraph()
  g.add_node(0, name='A')
  g.add_node(1, name='B')
  g.add_edge(0, 1)
  assert g.IsValidControlFlowGraph()


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
