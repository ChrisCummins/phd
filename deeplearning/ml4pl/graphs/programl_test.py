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
"""Unit tests for //deeplearning/ml4pl/graphs:programl."""
import random

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import decorators
from labm8.py import test

pytest_plugins = ["deeplearning.ml4pl.testing.fixtures.llvm_program_graph"]

FLAGS = test.FLAGS

###############################################################################
# Fixtures.
###############################################################################


@test.Fixture(scope="session", params=(1, 2))
def node_x_dimensionality(request) -> int:
  """A test fixture which enumerates dimensionalities."""
  return request.param


@test.Fixture(scope="session", params=(0, 2))
def node_y_dimensionality(request) -> int:
  """A test fixture which enumerates dimensionalities."""
  return request.param


@test.Fixture(scope="session", params=(0, 2))
def graph_x_dimensionality(request) -> int:
  """A test fixture which enumerates dimensionalities."""
  return request.param


@test.Fixture(scope="session", params=(0, 2))
def graph_y_dimensionality(request) -> int:
  """A test fixture which enumerates dimensionalities."""
  return request.param


@test.Fixture(scope="session", params=(None, 10, 100))
def node_count(request) -> int:
  """A test fixture which enumerates node_counts."""
  return request.param


@test.Fixture(scope="session", params=list(programl.InputOutputFormat))
def fmt(request) -> programl.InputOutputFormat:
  """A test fixture which enumerates protocol buffer formats."""
  return request.param


###############################################################################
# Tests.
###############################################################################


def test_graphviz_converter(llvm_program_graph: programl_pb2.ProgramGraph):
  """Black-box test ProgramGraphToGraphviz()."""
  assert programl.ProgramGraphToGraphviz(llvm_program_graph)


def test_proto_networkx_equivalence(
  llvm_program_graph: programl_pb2.ProgramGraph,
):
  """Test proto -> networkx -> proto equivalence."""
  # proto -> networkx
  g = programl.ProgramGraphToNetworkX(llvm_program_graph)
  assert g.number_of_nodes() == len(llvm_program_graph.node)
  assert g.number_of_edges() == len(llvm_program_graph.edge)

  # networkx -> proto
  proto_out = programl.NetworkXToProgramGraph(g)
  assert set(fn.name for fn in proto_out.function) == set(
    fn.name for fn in llvm_program_graph.function
  )
  assert len(llvm_program_graph.node) == len(proto_out.node)
  assert len(llvm_program_graph.edge) == len(proto_out.edge)


def test_proto_networkx_equivalence_with_preallocated_proto(
  llvm_program_graph: programl_pb2.ProgramGraph,
):
  """Test proto -> networkx -> proto equivalent using pre-allocated protos."""
  # proto -> networkx
  g = programl.ProgramGraphToNetworkX(llvm_program_graph)
  assert g.number_of_nodes() == len(llvm_program_graph.node)
  assert g.number_of_edges() == len(llvm_program_graph.edge)

  # networkx -> proto
  # Allocate the proto ahead of time:
  proto_out = programl_pb2.ProgramGraph()
  programl.NetworkXToProgramGraph(g, proto=proto_out)
  assert set(fn.name for fn in proto_out.function) == set(
    fn.name for fn in llvm_program_graph.function
  )
  assert len(llvm_program_graph.node) == len(proto_out.node)
  assert len(llvm_program_graph.edge) == len(proto_out.edge)


###############################################################################
# Fuzzers.
###############################################################################


@decorators.loop_for(seconds=10)
def test_fuzz_GraphBuilder():
  """Test that graph construction doesn't set on fire."""
  builder = programl.GraphBuilder()
  random_node_count = random.randint(3, 100)
  random_edge_count = random.randint(3, 100)
  nodes = []
  for _ in range(random_node_count):
    nodes.append(builder.AddNode())
  for _ in range(random_edge_count):
    builder.AddEdge(random.choice(nodes), random.choice(nodes))
  assert builder.g
  assert builder.proto


@decorators.loop_for(seconds=3)
def test_fuzz_proto_bytes_equivalence(fmt: programl.InputOutputFormat):
  """Test that conversion to and from bytes does not change the proto."""
  input = random_programl_generator.CreateRandomProto()
  output = programl.FromBytes(programl.ToBytes(input, fmt), fmt)
  assert input == output


@decorators.loop_for(seconds=3)
def test_fuzz_proto_networkx_equivalence(
  node_x_dimensionality: int,
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
  node_count: int,
):
  """Fuzz proto -> networkx -> proto on random generated graphs."""
  proto_in = random_programl_generator.CreateRandomProto(
    node_x_dimensionality=node_x_dimensionality,
    node_y_dimensionality=node_y_dimensionality,
    graph_x_dimensionality=graph_x_dimensionality,
    graph_y_dimensionality=graph_y_dimensionality,
    node_count=node_count,
  )

  # proto -> networkx
  g = programl.ProgramGraphToNetworkX(proto_in)
  assert g.number_of_nodes() == len(proto_in.node)
  assert g.number_of_edges() == len(proto_in.edge)

  # Check that the functions match up.
  functions_in_graph = set(
    [
      function
      for _, function in g.nodes(data="function")
      if function is not None
    ]
  )
  functions_in_proto = [function.name for function in proto_in.function]
  assert sorted(functions_in_proto) == sorted(functions_in_graph)

  # networkx -> proto
  proto_out = programl.NetworkXToProgramGraph(g)
  assert proto_out.function == proto_in.function
  assert proto_out.node == proto_in.node
  # Randomly generated graphs don't have a stable edge order.
  assert len(proto_out.edge) == len(proto_in.edge)


if __name__ == "__main__":
  test.Main()
