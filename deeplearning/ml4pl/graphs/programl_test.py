"""Unit tests for //deeplearning/ml4pl/graphs:programl."""
import pickle
from typing import Iterable
from typing import Tuple

import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.migrate import networkx_to_protos
from deeplearning.ml4pl.graphs.unlabelled.cdfg import random_cdfg_generator
from labm8.py import bazelutil
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


NETWORKX_GRAPHS_ARCHIVE = bazelutil.DataArchive(
  "phd/deeplearning/ml4pl/testing/data/100_unlabelled_networkx_graphs.tar.bz2"
)


def ReadPickledNetworkxGraphs() -> Iterable[Tuple[str, nx.MultiDiGraph]]:
  """Read the pickled networkx graphs."""
  with NETWORKX_GRAPHS_ARCHIVE as pickled_dir:
    for path in pickled_dir.iterdir():
      with open(path, "rb") as f:
        yield path.name, pickle.load(f)


@test.Fixture(scope="function", params=list(ReadPickledNetworkxGraphs()))
def random_100_proto(request) -> programl_pb2.ProgramGraph:
  """A test fixture which returns one of 100 "real" graph protos."""
  name, g = request.param
  return networkx_to_protos.NetworkXGraphToProgramGraphProto(g)


def CreateRandomProto() -> programl_pb2.ProgramGraph:
  """Generate a random graph proto."""
  g = random_cdfg_generator.FastCreateRandom()
  return networkx_to_protos.NetworkXGraphToProgramGraphProto(g)


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


@decorators.loop_for(seconds=10)
def test_fuzz_proto_networkx_equivalence():
  """Fuzz proto -> networkx -> proto on random generated graphs."""
  proto_in = CreateRandomProto()

  # proto -> networkx
  g = programl.ProgramGraphToNetworkX(proto_in)
  assert g.number_of_nodes() == len(proto_in.node)
  assert g.number_of_edges() == len(proto_in.edge)

  # networkx -> proto
  proto_out = programl.NetworkXToProgramGraph(g)
  assert proto_out.function == proto_in.function
  assert proto_out.node == proto_in.node
  # Randomly generated graphs don't have a stable edge order.
  assert len(proto_out.edge) == len(proto_in.edge)


# Benchmarks:


@test.Fixture(scope="module")
def random_proto() -> programl_pb2.ProgramGraph:
  return CreateRandomProto()


def test_benchmark_proto_to_networkx(
  benchmark, random_proto: programl_pb2.ProgramGraph
):
  """Benchmark proto -> networkx."""
  benchmark(programl.ProgramGraphToNetworkX, random_proto)


def test_benchmark_networkx_to_proto(
  benchmark, random_proto: programl_pb2.ProgramGraph
):
  """Benchmark networkx -> proto."""
  g = programl.ProgramGraphToNetworkX(random_proto)
  benchmark(programl.NetworkXToProgramGraph, g)


if __name__ == "__main__":
  test.Main()
