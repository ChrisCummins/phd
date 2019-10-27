"""Unit tests for //deeplearning/ml4pl:control_and_data_flow_graph."""
import networkx as nx
import pytest
from labm8 import app
from labm8 import test

from deeplearning.ml4pl.graphs import graph_iterators as iterators
from deeplearning.ml4pl.graphs import graph_query as query
from deeplearning.ml4pl.graphs.unlabelled.cdfg import \
  control_and_data_flow_graph as cdfg

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def simple_bytecode() -> str:
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
  return """
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
"""


def test_that_root_node_is_connected_to_entry_points(simple_bytecode: str):
  builder = cdfg.ControlAndDataFlowGraphBuilder()
  graph = builder.Build(simple_bytecode)
  assert 'root' in graph
  assert graph.in_degree('root') == 0
  assert len(graph.out_edges('root')) == 2
  assert set(graph.neighbors('root')) == {'A_0', 'B_0'}


def test_every_statement_has_a_predecessor(simple_bytecode: str):
  """Test that every statement (except entry blocks) have control preds."""
  builder = cdfg.ControlAndDataFlowGraphBuilder()
  graph = builder.Build(simple_bytecode)
  entry_blocks = set([node for node, _ in iterators.EntryBlockIterator(graph)])
  for node, _ in iterators.StatementNodeIterator(graph):
    if node in entry_blocks:
      continue
    for edge in graph.in_edges(node):
      if graph.edges[edge[0], edge[1], 0]['flow'] == 'control':
        break
    else:
      assert False, f'{node} has no control flow predecessor.'


def test_StatementIsSuccessor_linear_control_path():
  g = nx.MultiDiGraph()
  g.add_edge('a', 'b', type='control')
  g.add_edge('b', 'c', type='control')
  assert query.StatementIsSuccessor(g, 'a', 'a')
  assert query.StatementIsSuccessor(g, 'a', 'b')
  assert query.StatementIsSuccessor(g, 'a', 'c')
  assert query.StatementIsSuccessor(g, 'b', 'c')
  assert not query.StatementIsSuccessor(g, 'c', 'a')
  assert not query.StatementIsSuccessor(g, 'b', 'a')
  assert not query.StatementIsSuccessor(g, 'a', '_not_in_graph_')
  with pytest.raises(Exception):
    assert not query.StatementIsSuccessor(g, '_not_in_graph_',
                                          '_not_in_graph2_')


def test_StatementIsSuccessor_branched_control_path():
  g = nx.MultiDiGraph()
  g.add_edge('a', 'b', type='control')
  g.add_edge('a', 'c', type='control')
  g.add_edge('b', 'd', type='control')
  g.add_edge('c', 'd', type='control')
  assert query.StatementIsSuccessor(g, 'a', 'b')
  assert query.StatementIsSuccessor(g, 'a', 'c')
  assert query.StatementIsSuccessor(g, 'a', 'b')
  assert not query.StatementIsSuccessor(g, 'b', 'a')
  assert not query.StatementIsSuccessor(g, 'b', 'c')
  assert query.StatementIsSuccessor(g, 'b', 'd')


if __name__ == '__main__':
  test.Main()
