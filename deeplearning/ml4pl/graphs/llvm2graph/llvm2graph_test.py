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
"""Unit tests for //deeplearning/ml4pl/graphs/llvm2graph:llvm2graph_py."""
from deeplearning.ml4pl.graphs import nx_utils
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.llvm2graph import llvm2graph
from labm8.py import app
from labm8.py import test


FLAGS = app.FLAGS

pytest_plugins = ["deeplearning.ml4pl.testing.fixtures.llvm_ir"]


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
  graph = llvm2graph.BuildProgramGraphNetworkX(simple_bytecode)
  assert graph.nodes[0]["text"] == "; root"
  # Call edges from A -> root, B -> root.
  assert graph.in_degree(0) == 2
  assert len(graph.out_edges(0)) == 2


def test_every_statement_has_a_predecessor(simple_bytecode: str):
  """Test that every statement (except entry blocks) have control preds."""
  graph = llvm2graph.BuildProgramGraphNetworkX(simple_bytecode)
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
  graph = llvm2graph.BuildProgramGraphNetworkX(simple_bytecode)
  for src, dst, position in graph.edges(data="position"):
    assert isinstance(
      position, int
    ), f'No position for edge {graph.nodes[src]["text"]} -> {graph.nodes[dst]["text"]}'


def test_every_node_has_x(simple_bytecode: str):
  """Test that every edge has a position encoding."""
  graph = llvm2graph.BuildProgramGraphNetworkX(simple_bytecode)
  for node, x in graph.nodes(data="x"):
    assert isinstance(
      x, list
    ), f"Invalid x attribute for node {graph.nodes[node]}"


def test_on_real_ir(llvm_ir: str):
  """Test the graph construction on real LLVM IR."""
  llvm2graph.BuildProgramGraphNetworkX(llvm_ir)


if __name__ == "__main__":
  test.Main()
