# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //deeplearning/ml4pl/graphs/unlabelled/llvm2graph:graph_builder."""
import networkx as nx

from deeplearning.ml4pl.graphs import nx_utils
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.unlabelled.llvm2graph import graph_builder
from labm8.py import app
from labm8.py import test


FLAGS = app.FLAGS


@test.Fixture(scope="function")
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
  builder = graph_builder.ProGraMLGraphBuilder()
  graph = builder.Build(simple_bytecode)
  assert graph.nodes[0]["text"] == "root"
  assert graph.in_degree(0) == 0
  assert len(graph.out_edges(0)) == 2


def test_every_statement_has_a_predecessor(simple_bytecode: str):
  """Test that every statement (except entry blocks) have control preds."""
  builder = graph_builder.ProGraMLGraphBuilder()
  graph = builder.Build(simple_bytecode)
  entry_blocks = set([node for node, _ in nx_utils.EntryBlockIterator(graph)])
  for node, _ in nx_utils.StatementNodeIterator(graph):
    if not node or node in entry_blocks:
      continue
    for edge in graph.in_edges(node):
      if graph.edges[edge[0], edge[1], 0]["flow"] == programl_pb2.Edge.CONTROL:
        break
    else:
      assert False, f"{node} has no control flow predecessor."


def test_every_edge_has_position(simple_bytecode: str):
  """Test that every edge has a position encoding."""
  builder = graph_builder.ProGraMLGraphBuilder()
  graph = builder.Build(simple_bytecode)
  for src, dst, position in graph.edges(data="position"):
    assert isinstance(
      position, int
    ), f'No position for edge {graph.nodes[src]["text"]} -> {graph.nodes[dst]["text"]}'


def test_every_node_has_x(simple_bytecode: str):
  """Test that every edge has a position encoding."""
  builder = graph_builder.ProGraMLGraphBuilder()
  graph = builder.Build(simple_bytecode)
  for node, x in graph.nodes(data="x"):
    assert isinstance(
      x, list
    ), f"Invalid x attribute for node {graph.nodes[node]}"


@test.XFail(reason="TODO(github.com/ChrisCummins/ProGraML/issues/2)")
def test_ComposeGraphs_undefined():
  """Test that function graph is inserted for call to undefined function."""
  builder = graph_builder.ProGraMLGraphBuilder()

  A = nx.MultiDiGraph(name="A")
  A.entry_block = "A_entry"
  A.exit_block = "A_exit"

  A.add_node(A.entry_block, type="statement", function="A", text="")
  A.add_node(A.exit_block, type="statement", function="A", text="")
  A.add_node("call", type="statement", function="A", text="call i32 @B(i32 1)")

  A.add_edge(A.entry_block, "call", flow="control", function="A")
  A.add_edge("call", A.exit_block, flow="control", function="A")

  call_graph = nx.MultiDiGraph()
  call_graph.add_edge("external node", "A")
  call_graph.add_edge("external node", "B")
  call_graph.add_edge("A", "B")

  g = builder.ComposeGraphs([A], call_graph)

  assert "root" in g
  assert "B_entry" in g
  assert "B_exit" in g

  assert g.edges("call", "B_entry")
  assert g.edges("B_exit", "call")

  assert g.number_of_nodes() == 6

  assert g.edges("root", "A")
  assert g.edges("root", "B")


@test.XFail(reason="TODO(github.com/ChrisCummins/ProGraML/issues/2)")
def test_FindCallSites_multiple_call_sites():
  g = nx.MultiDiGraph()
  g.add_node("call", type="statement", function="A", text="%2 = call i32 @B()")
  g.add_node("foo", type="statement", function="A", text="")
  g.add_node(
    "call2", type="statement", function="A", text="%call = call i32 @B()"
  )

  call_sites = graph_builder.FindCallSites(g, "A", "B")
  assert len(call_sites) == 2
  assert set(call_sites) == {"call", "call2"}


if __name__ == "__main__":
  test.Main()
