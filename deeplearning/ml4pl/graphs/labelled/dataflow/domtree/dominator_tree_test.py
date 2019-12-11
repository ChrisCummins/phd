"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow/domtree:dominator_tree."""
import random

import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow.domtree import dominator_tree
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS

###############################################################################
# Fixtures.
###############################################################################


@test.Fixture(scope="function")
def annotator() -> dominator_tree.DominatorTreeAnnotator:
  return dominator_tree.DominatorTreeAnnotator()


@test.Fixture(
  scope="session",
  params=list(random_programl_generator.EnumerateProtoTestSet()),
)
def real_graph(request) -> programl_pb2.ProgramGraph:
  """A test fixture which yields one of 100 "real" graphs."""
  return request.param


@test.Fixture(scope="function")
def g1() -> nx.MultiDiGraph:
  """A four statement graph with one data element."""
  #            +---+
  #   +--------+ a +--------+       +---+
  #   |        +---+        |       | v1|
  #   V                     v       +---+
  # +---+                 +-+-+       |
  # | b |                 | c +<------+
  # +-+-+                 +-+-+
  #   |        +---+        |
  #   +------->+ d +<-------+
  #            +---+
  builder = programl.GraphBuilder()
  a = builder.AddNode(x=[-1])
  b = builder.AddNode(x=[-1])
  c = builder.AddNode(x=[-1])
  d = builder.AddNode(x=[-1])
  v1 = builder.AddNode(x=[-1], type=programl_pb2.Node.IDENTIFIER)
  builder.AddEdge(a, b)
  builder.AddEdge(a, c)
  builder.AddEdge(b, d)
  builder.AddEdge(c, d)
  builder.AddEdge(v1, c, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(a, c, flow=programl_pb2.Edge.DATA)
  return builder.g


@test.Fixture(scope="function")
def g2() -> nx.MultiDiGraph:
  """A five statement graph with one data element.

  This is the same as g1, but the extra statement "e" means that 'a' no longer
  dominates all of the other nodes.
  """
  #                       +---+
  # +---+        +--------+ a +--------+       +---+
  # | e |        |        +---+        |       | v1|
  # +-+-+        V                     v       +---+
  #   |        +---+                 +-+-+       ||
  #   +------->+ b |                 | c +<======++
  #            +-+-+                 +-+-+
  #              |        +---+        |
  #              +------->+ d +<-------+
  #                       +---+
  #
  builder = programl.GraphBuilder()
  a = builder.AddNode(x=[-1])
  b = builder.AddNode(x=[-1])
  c = builder.AddNode(x=[-1])
  d = builder.AddNode(x=[-1])
  e = builder.AddNode(x=[-1])
  v1 = builder.AddNode(x=[-1], type=programl_pb2.Node.IDENTIFIER)
  builder.AddEdge(a, b)
  builder.AddEdge(a, c)
  builder.AddEdge(b, d)
  builder.AddEdge(c, d)
  builder.AddEdge(v1, c, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(a, c, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(e, b)
  return builder.g


###############################################################################
# Tests.
###############################################################################


def test_Annotate_g1(
  g1: nx.MultiDiGraph, annotator: dominator_tree.DominatorTreeAnnotator
):
  """Test dominator tree for a small """
  annotated = annotator.Annotate(g1, root_node=0)

  assert annotated.graph["data_flow_positive_node_count"] == 4
  assert annotated.graph["data_flow_steps"] == 2

  # Features
  assert annotated.nodes[0]["x"] == [-1, 1]
  assert annotated.nodes[1]["x"] == [-1, 0]
  assert annotated.nodes[2]["x"] == [-1, 0]
  assert annotated.nodes[3]["x"] == [-1, 0]
  assert annotated.nodes[4]["x"] == [-1, 0]

  # Labels
  assert annotated.nodes[0]["y"] == dominator_tree.DOMINATED
  assert annotated.nodes[1]["y"] == dominator_tree.DOMINATED
  assert annotated.nodes[2]["y"] == dominator_tree.DOMINATED
  assert annotated.nodes[3]["y"] == dominator_tree.DOMINATED
  assert annotated.nodes[4]["y"] == dominator_tree.NOT_DOMINATED


def test_Annotate_g2(
  g2: nx.MultiDiGraph, annotator: dominator_tree.DominatorTreeAnnotator
):
  """Test dominator tree for a small """
  annotated = annotator.Annotate(g2, root_node=0)

  assert annotated.graph["data_flow_positive_node_count"] == 2
  assert annotated.graph["data_flow_steps"] in {2, 3}

  # Features
  assert annotated.nodes[0]["x"] == [-1, 1]
  assert annotated.nodes[1]["x"] == [-1, 0]
  assert annotated.nodes[2]["x"] == [-1, 0]
  assert annotated.nodes[3]["x"] == [-1, 0]
  assert annotated.nodes[4]["x"] == [-1, 0]
  assert annotated.nodes[5]["x"] == [-1, 0]

  # Labels
  assert annotated.nodes[0]["y"] == dominator_tree.DOMINATED
  assert annotated.nodes[1]["y"] == dominator_tree.NOT_DOMINATED
  assert annotated.nodes[2]["y"] == dominator_tree.DOMINATED
  assert annotated.nodes[3]["y"] == dominator_tree.NOT_DOMINATED
  assert annotated.nodes[4]["y"] == dominator_tree.NOT_DOMINATED
  assert annotated.nodes[5]["y"] == dominator_tree.NOT_DOMINATED


def test_MakeAnnotated_real_graphs(
  real_graph: programl_pb2.ProgramGraph,
  annotator: dominator_tree.DominatorTreeAnnotator,
):
  """Opaque black-box test of reachability annotator."""
  annotated = annotator.MakeAnnotated(real_graph, n=10)
  assert len(annotated.graphs) <= 10


###############################################################################
# Fuzzers.
###############################################################################


@decorators.loop_for(seconds=30)
def test_fuzz_MakeAnnotated(annotator: dominator_tree.DominatorTreeAnnotator,):
  """Opaque black-box test of reachability annotator."""
  n = random.randint(1, 20)
  proto = random_programl_generator.CreateRandomProto()
  annotated = annotator.MakeAnnotated(proto, n=n)
  assert len(annotated.graphs) <= 20
  assert len(annotated.protos) <= 20


if __name__ == "__main__":
  test.Main()
