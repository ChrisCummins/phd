"""Unit tests for //deeplearning/ml4pl/graphs:programl."""
import random
from typing import List

import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS

###############################################################################
# Fixtures.
###############################################################################


@test.Fixture(
  scope="function",
  params=list(random_programl_generator.EnumerateProtoTestSet()),
)
def random_100_proto(request) -> programl_pb2.ProgramGraph:
  """A test fixture which returns one of 100 "real" graph protos."""
  return request.param


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


###############################################################################
# Tests.
###############################################################################


def test_proto_networkx_equivalence(
  random_100_proto: programl_pb2.ProgramGraph,
):
  """Test proto -> networkx -> proto on 100 "real" graphs."""
  # proto -> networkx
  g = programl.ProgramGraphToNetworkX(random_100_proto)
  assert g.number_of_nodes() == len(random_100_proto.node)
  assert g.number_of_edges() == len(random_100_proto.edge)

  # networkx -> proto
  proto_out = programl.NetworkXToProgramGraph(g)
  assert proto_out.function == random_100_proto.function
  assert proto_out.node == random_100_proto.node
  assert proto_out.edge == random_100_proto.edge


def test_proto_networkx_equivalence_with_preallocated_proto(
  random_100_proto: programl_pb2.ProgramGraph,
):
  """Test proto -> networkx -> proto on 100 "real" graphs using the same
  proto instance."""
  # proto -> networkx
  g = programl.ProgramGraphToNetworkX(random_100_proto)
  assert g.number_of_nodes() == len(random_100_proto.node)
  assert g.number_of_edges() == len(random_100_proto.edge)

  # networkx -> proto
  # Allocate the proto ahead of time:
  proto_out = programl_pb2.ProgramGraph()
  programl.NetworkXToProgramGraph(g, proto=proto_out)
  assert proto_out.function == random_100_proto.function
  assert proto_out.node == random_100_proto.node
  assert proto_out.edge == random_100_proto.edge


###############################################################################
# Fuzzers.
###############################################################################


@decorators.loop_for(seconds=30)
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


@decorators.loop_for(seconds=5)
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


###############################################################################
# Benchmarks.
###############################################################################


@test.Fixture(scope="session")
def benchmark_protos(
  node_x_dimensionality: int,
  node_y_dimensionality: int,
  graph_x_dimensionality: int,
  graph_y_dimensionality: int,
  node_count: int,
) -> List[programl_pb2.ProgramGraph]:
  """A fixture which returns 10 protos for benchmarking."""
  return [
    random_programl_generator.CreateRandomProto(
      node_x_dimensionality=node_x_dimensionality,
      node_y_dimensionality=node_y_dimensionality,
      graph_x_dimensionality=graph_x_dimensionality,
      graph_y_dimensionality=graph_y_dimensionality,
      node_count=node_count,
    )
    for _ in range(10)
  ]


@test.Fixture(scope="session")
def benchmark_networkx(
  benchmark_protos: List[programl_pb2.ProgramGraph],
) -> List[nx.MultiDiGraph]:
  """A fixture which returns 10 graphs for benchmarking."""
  return [programl.ProgramGraphToNetworkX(p) for p in benchmark_protos]


def Benchmark(fn, inputs):
  """A micro-benchmark which calls the given function over all inputs."""
  for element in inputs:
    fn(element)


def test_benchmark_proto_to_networkx(
  benchmark, benchmark_protos: List[programl_pb2.ProgramGraph]
):
  """Benchmark proto -> networkx."""
  benchmark(Benchmark, programl.ProgramGraphToNetworkX, benchmark_protos)


def test_benchmark_networkx_to_proto(
  benchmark, benchmark_networkx: List[nx.MultiDiGraph]
):
  """Benchmark networkx -> proto."""
  benchmark(Benchmark, programl.NetworkXToProgramGraph, benchmark_networkx)


if __name__ == "__main__":
  test.Main()
