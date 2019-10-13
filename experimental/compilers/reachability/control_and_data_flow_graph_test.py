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
  # assert len(graph1.out_edges('root')) == 1
  assert list(graph1.neighbors('root')) == []


def test_every_statement_has_a_predecessor(graph1: nx.MultiDiGraph):
  for node, _ in cdfg.StatementNodeIterator(graph1):
    for edge in graph1.in_edges(node):
      if graph1.edges[edge[0], edge[1], 0]['flow'] == 'control':
        break
    else:
      assert False, f'{node} has no control flow predecessor.'


if __name__ == '__main__':
  test.Main()
