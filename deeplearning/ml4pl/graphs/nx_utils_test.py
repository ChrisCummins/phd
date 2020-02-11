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
"""Unit tests for //deeplearning/ml4pl/graphs:nx_utils."""
from deeplearning.ml4pl.graphs import nx_utils
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.py import graph_builder
from labm8.py import test

FLAGS = test.FLAGS


@test.Fixture(scope="function")
def graph():
  builder = graph_builder.GraphBuilder()
  fn = builder.AddFunction("x")
  A = builder.AddStatement("x", fn)
  B = builder.AddStatement("x", fn)
  C = builder.AddStatement("x", fn)
  D = builder.AddStatement("x", fn)
  v1 = builder.AddIdentifier("x", fn)
  builder.AddControlEdge(A, B)
  builder.AddControlEdge(B, C)
  builder.AddControlEdge(C, D)
  builder.AddCallEdge(0, A)
  builder.AddDataEdge(A, v1)
  builder.AddDataEdge(v1, D)
  return builder.g


def test_StatementNodeIterator(graph):
  assert len(list(nx_utils.StatementNodeIterator(graph))) == 5


def test_IdentifierNodeIterator(graph):
  assert len(list(nx_utils.IdentifierNodeIterator(graph))) == 1


def test_EntryBlockIterator(graph):
  assert len(list(nx_utils.EntryBlockIterator(graph))) == 1


def test_ExitBlockIterator(graph):
  assert len(list(nx_utils.ExitBlockIterator(graph))) == 1


def test_ControlFlowEdgeIterator(graph):
  assert len(list(nx_utils.ControlFlowEdgeIterator(graph))) == 3


def test_DataFlowEdgeIterator(graph):
  assert len(list(nx_utils.DataFlowEdgeIterator(graph))) == 2


def test_StatementNeighbors(graph):
  assert nx_utils.StatementNeighbors(graph, 1) == {2}


def test_StatementNeighbors_data_flow(graph):
  assert nx_utils.StatementNeighbors(graph, 1, flow=programl_pb2.Edge.DATA) == {
    4
  }


def test_StatementIsSuccessor(graph):
  assert nx_utils.StatementIsSuccessor(graph, 1, 2)
  assert not nx_utils.StatementIsSuccessor(graph, 2, 1)


def test_StatementIsSuccessor_linear_control_path():
  # A -> B -> C
  builder = graph_builder.GraphBuilder()
  fn = builder.AddFunction("x")
  a = builder.AddStatement("a", fn)
  b = builder.AddStatement("a", fn)
  c = builder.AddStatement("a", fn)
  builder.AddControlEdge(0, a)
  builder.AddControlEdge(a, b)
  builder.AddControlEdge(b, c)
  g = builder.g
  assert nx_utils.StatementIsSuccessor(g, a, a)
  assert nx_utils.StatementIsSuccessor(g, a, b)
  assert nx_utils.StatementIsSuccessor(g, a, c)
  assert nx_utils.StatementIsSuccessor(g, b, c)
  assert not nx_utils.StatementIsSuccessor(g, c, a)
  assert not nx_utils.StatementIsSuccessor(g, b, a)
  assert not nx_utils.StatementIsSuccessor(g, a, -1)  # Destination not in graph
  with test.Raises(Exception):
    nx_utils.StatementIsSuccessor(g, -1, a)  # Source not in graph


def test_StatementIsSuccessor_branched_control_path():
  # A -> B -> D
  # A -> C -> D
  builder = graph_builder.GraphBuilder()
  fn = builder.AddFunction("x")
  a = builder.AddStatement("A", fn)
  b = builder.AddStatement("B", fn)
  c = builder.AddStatement("C", fn)
  d = builder.AddStatement("D", fn)
  builder.AddControlEdge(0, a)
  builder.AddControlEdge(a, b)
  builder.AddControlEdge(a, c)
  builder.AddControlEdge(b, d)
  builder.AddControlEdge(c, d)
  g = builder.g
  assert nx_utils.StatementIsSuccessor(g, a, b)
  assert nx_utils.StatementIsSuccessor(g, a, c)
  assert nx_utils.StatementIsSuccessor(g, a, b)
  assert not nx_utils.StatementIsSuccessor(g, b, a)
  assert not nx_utils.StatementIsSuccessor(g, b, c)
  assert nx_utils.StatementIsSuccessor(g, b, d)


def test_GetStatementsForNode_node():
  """Test the nodes returned when root is a statementt."""
  builder = graph_builder.GraphBuilder()
  fn = builder.AddFunction("x")
  foo = builder.AddStatement("x", fn)
  builder.AddControlEdge(0, foo)

  nodes = list(nx_utils.GetStatementsForNode(builder.g, foo))
  assert nodes == [foo]


def test_GetStatementsForNode_identifier():
  """Test the nodes returned when root is an identifier."""
  builder = graph_builder.GraphBuilder()
  fn = builder.AddFunction("x")
  foo = builder.AddStatement("x", fn)
  bar = builder.AddStatement("x", fn)
  v1 = builder.AddIdentifier("v1", fn)
  builder.AddControlEdge(0, foo)
  builder.AddDataEdge(foo, v1, 0)
  builder.AddDataEdge(v1, bar, 0)

  nodes = list(nx_utils.GetStatementsForNode(builder.g, v1))
  assert nodes == [foo, bar]


if __name__ == "__main__":
  test.Main()
