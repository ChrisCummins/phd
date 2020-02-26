# Copyright 2019-2020 the ProGraML authors.
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
"""Unit tests for //deeplearning/ml4pl/graphs/llvm2graph:node_encoder."""
import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.llvm2graph import node_encoder
from labm8.py import test


FLAGS = test.FLAGS

pytest_plugins = ["deeplearning.ml4pl.testing.fixtures.llvm_program_graph"]


@test.Fixture(scope="session")
def encoder() -> node_encoder.GraphNodeEncoder:
  """A session-level fixture to re-use a graph encoder instance."""
  return node_encoder.GraphNodeEncoder()


def test_EncodeNodes_equivalent_preprocessed_text(
  encoder: node_encoder.GraphNodeEncoder,
):
  """Test equivalence of nodes that pre-process to the same text."""
  builder = programl.GraphBuilder()
  a = builder.AddNode(text="%7 = add nsw i32 %5, -1")
  b = builder.AddNode(text="%9 = add nsw i32 %5, -2")
  g = builder.g

  encoder.EncodeNodes(g)

  assert g.nodes[a]["preprocessed_text"] == "<%ID> = add nsw i32 <%ID>, <INT>"
  assert g.nodes[b]["preprocessed_text"] == "<%ID> = add nsw i32 <%ID>, <INT>"


def test_EncodeNodes_identifier(encoder: node_encoder.GraphNodeEncoder):
  """Test the encoding of identifier nodes."""
  builder = programl.GraphBuilder()
  a = builder.AddNode(text="abcd", type=programl_pb2.Node.IDENTIFIER)
  g = builder.g

  encoder.EncodeNodes(g)
  assert g.nodes[a]["text"] == "abcd"
  assert g.nodes[a]["preprocessed_text"] == "!IDENTIFIER"


def test_EncodeNodes_immediate(encoder: node_encoder.GraphNodeEncoder):
  """Test the encoding of immediate nodes."""
  builder = programl.GraphBuilder()
  a = builder.AddNode(text="abcd", type=programl_pb2.Node.IMMEDIATE)
  g = builder.g

  encoder.EncodeNodes(g)
  assert g.nodes[a]["text"] == "abcd"
  assert g.nodes[a]["preprocessed_text"] == "!IMMEDIATE"


def test_EncodeNodes_encoded_values(encoder: node_encoder.GraphNodeEncoder):
  """Test that "x" attribute of a node matches dictionary value."""
  builder = programl.GraphBuilder()
  a = builder.AddNode(text="br label %4")
  g = builder.g

  encoder.EncodeNodes(g)

  assert g.nodes[a]["x"][0] == encoder.dictionary["br label <%ID>"]


def test_EncodeNodes_encoded_values_differ_between_statements(
  encoder: node_encoder.GraphNodeEncoder,
):
  """Test that "x" attribute of nodes differ between different texts."""
  builder = programl.GraphBuilder()
  a = builder.AddNode(text="%7 = add nsw i32 %5, -1")
  b = builder.AddNode(text="br label %4")
  g = builder.g

  encoder.EncodeNodes(g)

  assert g.nodes[a]["x"][0] != g.nodes[b]["x"][0]


def test_EncodeNodes_inlined_struct(encoder: node_encoder.GraphNodeEncoder):
  """Test that struct definition is inlined in pre-processed text.

  NOTE(github.com/ChrisCummins/ProGraML/issues/57): Regression test.
  """
  ir = """
%struct.foo = type { i32, i8* }

define i32 @A(%struct.foo*) #0 {
  %2 = alloca %struct.foo*, align 8
  store %struct.foo* %0, %struct.foo** %2, align 8
  %3 = load %struct.foo*, %struct.foo** %2, align 8
  %4 = getelementptr inbounds %struct.foo, %struct.foo* %3, i32 0, i32 0
  %5 = load i32, i32* %4, align 8
  ret i32 %5
}
"""
  builder = programl.GraphBuilder()
  a = builder.AddNode(text="%2 = alloca %struct.foo*, align 8")
  g = builder.g

  encoder.EncodeNodes(g, ir=ir)

  assert (
    g.nodes[a]["preprocessed_text"] == "<%ID> = alloca { i32, i8* }*, align 8"
  )


def test_EncodeNodes_inlined_nested_struct(
  encoder: node_encoder.GraphNodeEncoder,
):
  """Test that nested struct definitions are inlined in pre-processed text.

  NOTE(github.com/ChrisCummins/ProGraML/issues/57): Regression test.
  """
  ir = """
%struct.bar = type { %struct.foo* }
%struct.foo = type { i32 }

; Function Attrs: noinline nounwind optnone uwtable
define i32 @Foo(%struct.bar*) #0 {
  %2 = alloca %struct.bar*, align 8
  store %struct.bar* %0, %struct.bar** %2, align 8
  %3 = load %struct.bar*, %struct.bar** %2, align 8
  %4 = getelementptr inbounds %struct.bar, %struct.bar* %3, i32 0, i32 0
  %5 = load %struct.foo*, %struct.foo** %4, align 8
  %6 = getelementptr inbounds %struct.foo, %struct.foo* %5, i32 0, i32 0
  %7 = load i32, i32* %6, align 4
  ret i32 %7
}
"""
  builder = programl.GraphBuilder()
  a = builder.AddNode(text="%2 = alloca %struct.bar*, align 8")
  b = builder.AddNode(text="store %struct.bar* %0, %struct.bar** %2, align 8")
  c = builder.AddNode(text="%3 = load %struct.bar*, %struct.bar** %2, align 8")
  d = builder.AddNode(
    text="%4 = getelementptr inbounds %struct.bar, %struct.bar* %3, i32 0, i32 0"
  )
  e = builder.AddNode(text="%5 = load %struct.foo*, %struct.foo** %4, align 8")
  f = builder.AddNode(
    text="%6 = getelementptr inbounds %struct.foo, %struct.foo* %5, i32 0, i32 0"
  )
  g = builder.g

  encoder.EncodeNodes(g, ir=ir)

  assert (
    g.nodes[a]["preprocessed_text"] == "<%ID> = alloca { { i32 }* }*, align 8"
  )
  assert (
    g.nodes[b]["preprocessed_text"]
    == "store { { i32 }* }* <%ID>, { { i32 }* }** <%ID>, align 8"
  )
  assert (
    g.nodes[c]["preprocessed_text"]
    == "<%ID> = load { { i32 }* }*, { { i32 }* }** <%ID>, align 8"
  )
  assert (
    g.nodes[d]["preprocessed_text"]
    == "<%ID> = getelementptr inbounds { { i32 }* }, { { i32 }* }* <%ID>, i32 <INT>, i32 <INT>"
  )
  assert (
    g.nodes[e]["preprocessed_text"]
    == "<%ID> = load { i32 }*, { i32 }** <%ID>, align 8"
  )
  assert (
    g.nodes[f]["preprocessed_text"]
    == "<%ID> = getelementptr inbounds { i32 }, { i32 }* <%ID>, i32 <INT>, i32 <INT>"
  )


def test_EncodeNodes_llvm_program_graph(llvm_program_graph_nx: nx.MultiDiGraph):
  """Black-box test encoding LLVM program graphs."""
  encoder = node_encoder.GraphNodeEncoder()
  g = llvm_program_graph_nx.copy()
  encoder.EncodeNodes(g)

  # This assumes that all of the test graphs have at least one statement.
  num_statements = sum(
    1 if data["type"] == programl_pb2.Node.STATEMENT else 0
    for _, data in g.nodes(data=True)
  )
  assert num_statements >= 1

  # Check for the presence of expected node attributes.
  for _, data in g.nodes(data=True):
    assert len(data["x"]) == 1
    assert len(data["y"]) == 0
    assert "preprocessed_text" in data


if __name__ == "__main__":
  test.Main()
