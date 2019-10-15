"""Unit tests for //experimental/compilers/reachability:control_and_data_flow_graph."""
import networkx as nx
import pytest

from experimental.compilers.reachability import \
  control_and_data_flow_graph as cdfg
from labm8 import app
from labm8 import test


FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def builder() -> cdfg.ControlAndDataFlowGraphBuilder:
  return cdfg.ControlAndDataFlowGraphBuilder()


@pytest.fixture(scope='function')
def graph1(builder: cdfg.ControlAndDataFlowGraphBuilder):
  """From C code:

      int B() {
        return 10;
      }

      int A() {
        int x = B();
        if (x == 5) {
          x += 1;
        }
        return x;
      }
  """
  return builder.Build("""
; Function Attrs: noinline nounwind optnone ssp uwtable
define i32 @B() #0 {
  ret i32 10
}

; Function Attrs: noinline nounwind optnone ssp uwtable
define i32 @A() #0 {
  %1 = alloca i32, align 4
  %2 = call i32 @B()
  store i32 %2, i32* %1, align 4
  %3 = load i32, i32* %1, align 4
  %4 = icmp eq i32 %3, 5
  br i1 %4, label %5, label %8

; <label>:5:                                      ; preds = %0
  %6 = load i32, i32* %1, align 4
  %7 = add nsw i32 %6, 1
  store i32 %7, i32* %1, align 4
  br label %8

; <label>:8:                                      ; preds = %5, %0
  %9 = load i32, i32* %1, align 4
  ret i32 %9
}
""")


def test_that_root_node_is_connected_to_entry_point(graph1: nx.MultiDiGraph):
  assert 'root' in graph1
  assert graph1.in_degree('root') == 0
  assert len(graph1.out_edges('root')) == 1
  assert list(graph1.neighbors('root')) == ['A_0']


def test_every_statement_has_a_predecessor(graph1: nx.MultiDiGraph):
  for node, _ in cdfg.StatementNodeIterator(graph1):
    for edge in graph1.in_edges(node):
      if graph1.edges[edge[0], edge[1], 0]['flow'] == 'control':
        break
    else:
      assert False, f'{node} has no control flow predecessor.'


def test_StatementIsSuccessor_linear_control_path():
  g = nx.MultiDiGraph()
  g.add_edge('a', 'b')
  g.add_edge('b', 'c')
  assert cdfg.StatementIsSuccessor(g, 'a', 'a')
  assert cdfg.StatementIsSuccessor(g, 'a', 'b')
  assert cdfg.StatementIsSuccessor(g, 'a', 'c')
  assert cdfg.StatementIsSuccessor(g, 'b', 'c')
  assert not cdfg.StatementIsSuccessor(g, 'c', 'a')
  assert not cdfg.StatementIsSuccessor(g, 'b', 'a')
  assert not cdfg.StatementIsSuccessor(g, 'a', '_not_in_graph_')
  with pytest.raises(Exception):
    assert not cdfg.StatementIsSuccessor(g, '_not_in_graph_', '_not_in_graph2_')


def test_StatementIsSuccessor_branched_control_path():
  g = nx.MultiDiGraph()
  g.add_edge('a', 'b')
  g.add_edge('a', 'c')
  g.add_edge('b', 'd')
  g.add_edge('c', 'd')
  assert cdfg.StatementIsSuccessor(g, 'a', 'b')
  assert cdfg.StatementIsSuccessor(g, 'a', 'c')
  assert cdfg.StatementIsSuccessor(g, 'a', 'b')
  assert not cdfg.StatementIsSuccessor(g, 'b', 'a')
  assert not cdfg.StatementIsSuccessor(g, 'b', 'c')
  assert cdfg.StatementIsSuccessor(g, 'b', 'd')


if __name__ == '__main__':
  test.Main()
