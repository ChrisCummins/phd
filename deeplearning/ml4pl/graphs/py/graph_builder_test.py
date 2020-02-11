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
"""Unit tests for //deeplearning/ml4pl/graphs/py:graph_builder."""
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.py import graph_builder
from labm8.py import test

FLAGS = test.FLAGS


def test_empty_proto():
  builder = graph_builder.GraphBuilder()
  assert builder.GetSerializedGraphProto()


def test_add_function():
  builder = graph_builder.GraphBuilder()
  foo = builder.AddFunction("foo")
  bar = builder.AddFunction("bar")

  assert foo == 0
  assert bar == 1

  proto = programl_pb2.ProgramGraphProto()
  proto.ParseFromString(builder.GetSerializedGraphProto())

  assert proto.string[proto.function[0].name] == "foo"
  assert proto.string[proto.function[1].name] == "bar"


def test_function_with_empty_name():
  builder = graph_builder.GraphBuilder()
  with test.Raises(ValueError) as e_ctx:
    builder.AddFunction("")

  assert str(e_ctx.value) == "Empty function name is invalid"


def test_graph_with_unconnected_node():
  builder = graph_builder.GraphBuilder()
  fn = builder.AddFunction("x")
  builder.AddStatement("x", fn)
  with test.Raises(ValueError) as e_ctx:
    builder.GetSerializedGraphProto()
  assert "Graph contains node with no connections" in str(e_ctx.value)


def test_linear_statement_control_flow():
  """Test that graph construction doesn't set on fire."""
  builder = graph_builder.GraphBuilder()
  fn = builder.AddFunction("x")
  a = builder.AddStatement("x", fn)
  b = builder.AddStatement("x", fn)
  builder.AddControlEdge(a, b)

  assert builder.g
  assert builder.proto


if __name__ == "__main__":
  test.Main()
