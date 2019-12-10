"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow/reachability."""
import pickle
from typing import Iterable

import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow.reachability import (
  reachability,
)
from deeplearning.ml4pl.graphs.migrate import networkx_to_protos
from labm8.py import bazelutil
from labm8.py import test

FLAGS = test.FLAGS


NETWORKX_GRAPHS_ARCHIVE = bazelutil.DataArchive(
  "phd/deeplearning/ml4pl/testing/data/100_unlabelled_networkx_graphs.tar.bz2"
)


@test.Fixture(scope="function")
def graph() -> nx.MultiDiGraph:
  g = nx.MultiDiGraph()
  g.add_node(0, type=programl_pb2.Node.STATEMENT, x=[0])
  g.add_node(1, type=programl_pb2.Node.STATEMENT, x=[0])
  g.add_node(2, type=programl_pb2.Node.STATEMENT, x=[0])
  g.add_node(3, type=programl_pb2.Node.STATEMENT, x=[0])
  g.add_edge(0, 1, flow=programl_pb2.Edge.CONTROL)
  g.add_edge(1, 2, flow=programl_pb2.Edge.CONTROL)
  g.add_edge(2, 3, flow=programl_pb2.Edge.CONTROL)
  return g


@test.Fixture(scope="function")
def annotator() -> reachability.ReachabilityAnnotator:
  return reachability.ReachabilityAnnotator()


def test_Annotate_reachable_node_count_D(
  graph: nx.MultiDiGraph, annotator: reachability.ReachabilityAnnotator
):
  annotated = annotator.Annotate(graph, 3)
  assert annotated.positive_node_count == 1


def test_Annotate_reachable_node_count_A(
  graph: nx.MultiDiGraph, annotator: reachability.ReachabilityAnnotator
):
  annotated = annotator.Annotate(graph, 0)
  assert annotated.positive_node_count == 4


def test_Annotate_data_flow_steps_D(
  graph: nx.MultiDiGraph, annotator: reachability.ReachabilityAnnotator
):
  annotated = annotator.Annotate(graph, 3)
  assert annotated.data_flow_steps == 1


def test_Annotate_data_flow_steps_A(
  graph: nx.MultiDiGraph, annotator: reachability.ReachabilityAnnotator
):
  annotated = annotator.Annotate(graph, 0)
  assert annotated.data_flow_steps == 4


def test_Annotate_node_x(
  graph: nx.MultiDiGraph, annotator: reachability.ReachabilityAnnotator
):
  annotated = annotator.Annotate(graph, 0)
  assert annotated.g.nodes[0]["x"] == [0, 1]
  assert annotated.g.nodes[1]["x"] == [0, 0]
  assert annotated.g.nodes[2]["x"] == [0, 0]
  assert annotated.g.nodes[3]["x"] == [0, 0]


def test_Annotate_node_y(
  graph: nx.MultiDiGraph, annotator: reachability.ReachabilityAnnotator
):
  annotated = annotator.Annotate(graph, 1)
  assert annotated.g.nodes[0]["y"] == [1, 0]
  assert annotated.g.nodes[1]["y"] == [0, 1]
  assert annotated.g.nodes[2]["y"] == [0, 1]
  assert annotated.g.nodes[3]["y"] == [0, 1]


def ReadPickledNetworkxGraphs() -> Iterable[nx.MultiDiGraph]:
  """Read the pickled networkx graphs."""
  with NETWORKX_GRAPHS_ARCHIVE as pickled_dir:
    for path in pickled_dir.iterdir():
      with open(path, "rb") as f:
        yield pickle.load(f)


@test.Fixture(scope="function", params=list(ReadPickledNetworkxGraphs()))
def random_100_graph(request) -> nx.MultiDiGraph:
  """A test fixture which yields one of 100 "real" graphs."""
  original_graph = request.param
  proto = networkx_to_protos.NetworkXGraphToProgramGraphProto(original_graph)
  return programl.ProgramGraphToNetworkX(proto)


def test_Annotate_random_100(
  random_100_graph: nx.MultiDiGraph,
  annotator: reachability.ReachabilityAnnotator,
):
  """Opaque black-box test of reachability annotator."""
  list(annotator.MakeAnnotated(random_100_graph, n=100))


if __name__ == "__main__":
  test.Main()
