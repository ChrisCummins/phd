"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow/liveness."""
import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow.liveness import liveness
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import test

FLAGS = test.FLAGS


###############################################################################
# Fixtures.
###############################################################################


@test.Fixture(
  scope="session", params=list(random_programl_generator.EnumerateTestSet()),
)
def real_graph(request) -> programl_pb2.ProgramGraph:
  """A test fixture which yields one of 100 "real" graphs."""
  return request.param


###############################################################################
# Graph test.
###############################################################################


@test.Fixture(scope="function")
def wiki() -> programl_pb2.ProgramGraph:
  """A test fixture which yields a program graph."""
  # Example graph taken from Wikipedia:
  # <https://en.wikipedia.org/wiki/Live_variable_analysis>
  #
  # // in: {}
  # b1: a = 3;
  #     b = 5;
  #     d = 4;
  #     x = 100; //x is never being used later thus not in the out set {a,b,d}
  #     if a > b then
  # // out: {a,b,d}    //union of all (in) successors of b1 => b2: {a,b}, and b3:{b,d}
  #
  # // in: {a,b}
  # b2: c = a + b;
  #     d = 2;
  # // out: {b,d}
  #
  # // in: {b,d}
  # b3: endif
  #     c = 4;
  #     return b * d + c;
  # // out:{}
  builder = programl.GraphBuilder()

  # Variables:
  a = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])  # 0
  b = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])  # 1
  c = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])  # 2
  d = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])  # 3
  x = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])  # 4

  # b1
  b1 = builder.AddNode(type=programl_pb2.Node.STATEMENT, x=[-1])  # 5
  # Defs
  builder.AddEdge(b1, a, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(b1, b, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(b1, d, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(b1, x, flow=programl_pb2.Edge.DATA)

  # b2
  b2 = builder.AddNode(type=programl_pb2.Node.STATEMENT, x=[-1])  # 6
  builder.AddEdge(b1, b2, flow=programl_pb2.Edge.CONTROL)
  # Defs
  builder.AddEdge(b2, c, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(b2, d, flow=programl_pb2.Edge.DATA)
  # Uses
  builder.AddEdge(a, b2, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(b, b2, flow=programl_pb2.Edge.DATA)

  b3a = builder.AddNode(type=programl_pb2.Node.STATEMENT, x=[-1])  # 7
  builder.AddEdge(b1, b3a, flow=programl_pb2.Edge.CONTROL)
  builder.AddEdge(b2, b3a, flow=programl_pb2.Edge.CONTROL)
  # Defs
  builder.AddEdge(b3a, c, flow=programl_pb2.Edge.DATA)

  b3b = builder.AddNode(type=programl_pb2.Node.STATEMENT, x=[-1])  # 8
  builder.AddEdge(b3a, b3b, flow=programl_pb2.Edge.CONTROL)
  # Uses
  builder.AddEdge(b, b3b, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(d, b3b, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(c, b3b, flow=programl_pb2.Edge.DATA)
  return builder.proto


def test_AnnotateLiveness_wiki_b1(wiki: programl_pb2.ProgramGraph):
  """Test liveness annotations from block b1."""
  annotator = liveness.LivenessAnnotator(wiki)
  g = annotator.g

  annotator.Annotate(g, 5)
  assert g.graph["data_flow_positive_node_count"] == 3
  assert g.graph["data_flow_steps"] == 5

  # Features:
  assert g.nodes[5]["x"] == [-1, 1]
  assert g.nodes[6]["x"] == [-1, 0]
  assert g.nodes[7]["x"] == [-1, 0]
  assert g.nodes[8]["x"] == [-1, 0]

  assert g.nodes[0]["x"] == [-1, 0]
  assert g.nodes[1]["x"] == [-1, 0]
  assert g.nodes[2]["x"] == [-1, 0]
  assert g.nodes[3]["x"] == [-1, 0]

  # Labels:
  assert g.nodes[5]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[6]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[7]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[8]["y"] == liveness.NOT_LIVE_OUT

  assert g.nodes[0]["y"] == liveness.LIVE_OUT
  assert g.nodes[1]["y"] == liveness.LIVE_OUT
  assert g.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[3]["y"] == liveness.LIVE_OUT


def test_AnnotateLiveness_wiki_b2(wiki: programl_pb2.ProgramGraph):
  """Test liveness annotations from block b2."""
  annotator = liveness.LivenessAnnotator(wiki)
  g = annotator.g

  annotator.Annotate(g, 6)
  assert g.graph["data_flow_positive_node_count"] == 2
  assert g.graph["data_flow_steps"] == 5

  # Features:
  assert g.nodes[5]["x"] == [-1, 0]
  assert g.nodes[6]["x"] == [-1, 1]
  assert g.nodes[7]["x"] == [-1, 0]
  assert g.nodes[8]["x"] == [-1, 0]

  assert g.nodes[0]["x"] == [-1, 0]
  assert g.nodes[1]["x"] == [-1, 0]
  assert g.nodes[2]["x"] == [-1, 0]
  assert g.nodes[3]["x"] == [-1, 0]

  # Labels:
  assert g.nodes[5]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[6]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[7]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[8]["y"] == liveness.NOT_LIVE_OUT

  assert g.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[1]["y"] == liveness.LIVE_OUT
  assert g.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[3]["y"] == liveness.LIVE_OUT


def test_AnnotateLiveness_wiki_b3a(wiki: programl_pb2.ProgramGraph):
  """Test liveness annotations from block b3a."""
  annotator = liveness.LivenessAnnotator(wiki)
  g = annotator.g

  annotator.Annotate(g, 7)
  assert g.graph["data_flow_positive_node_count"] == 3
  assert g.graph["data_flow_steps"] == 5

  # Features:
  assert g.nodes[5]["x"] == [-1, 0]
  assert g.nodes[6]["x"] == [-1, 0]
  assert g.nodes[7]["x"] == [-1, 1]
  assert g.nodes[8]["x"] == [-1, 0]

  assert g.nodes[0]["x"] == [-1, 0]
  assert g.nodes[1]["x"] == [-1, 0]
  assert g.nodes[2]["x"] == [-1, 0]
  assert g.nodes[3]["x"] == [-1, 0]

  # Labels:
  assert g.nodes[5]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[6]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[7]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[8]["y"] == liveness.NOT_LIVE_OUT

  assert g.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[1]["y"] == liveness.LIVE_OUT
  assert g.nodes[2]["y"] == liveness.LIVE_OUT
  assert g.nodes[3]["y"] == liveness.LIVE_OUT


def test_AnnotateLiveness_wiki_b3b(wiki: programl_pb2.ProgramGraph):
  """Test liveness annotations from block b3b."""
  annotator = liveness.LivenessAnnotator(wiki)
  g = annotator.g

  annotator.Annotate(g, 8)
  assert g.graph["data_flow_positive_node_count"] == 0
  assert g.graph["data_flow_steps"] == 5

  # Features:
  assert g.nodes[5]["x"] == [-1, 0]
  assert g.nodes[6]["x"] == [-1, 0]
  assert g.nodes[7]["x"] == [-1, 0]
  assert g.nodes[8]["x"] == [-1, 1]

  assert g.nodes[0]["x"] == [-1, 0]
  assert g.nodes[1]["x"] == [-1, 0]
  assert g.nodes[2]["x"] == [-1, 0]
  assert g.nodes[3]["x"] == [-1, 0]

  # Labels:
  assert g.nodes[5]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[6]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[7]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[8]["y"] == liveness.NOT_LIVE_OUT

  assert g.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[3]["y"] == liveness.NOT_LIVE_OUT


###############################################################################
# Graph test.
###############################################################################


@test.Fixture(scope="function")
def graph() -> programl_pb2.ProgramGraph:
  """A test fixture which yields a program graph."""
  builder = programl.GraphBuilder()
  a = builder.AddNode(type=programl_pb2.Node.STATEMENT, x=[-1])
  b = builder.AddNode(type=programl_pb2.Node.STATEMENT, x=[-1])
  c = builder.AddNode(type=programl_pb2.Node.STATEMENT, x=[-1])
  d = builder.AddNode(type=programl_pb2.Node.STATEMENT, x=[-1])

  v1 = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  v2 = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])

  builder.AddEdge(a, b, flow=programl_pb2.Edge.CONTROL)
  builder.AddEdge(a, c, flow=programl_pb2.Edge.CONTROL)
  builder.AddEdge(b, d, flow=programl_pb2.Edge.CONTROL)
  builder.AddEdge(c, d, flow=programl_pb2.Edge.CONTROL)

  builder.AddEdge(a, v1, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(v1, b, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(b, v2, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(v2, d, flow=programl_pb2.Edge.DATA)

  return builder.proto


def test_AnnotateDominatorTree_graph_A(graph: programl_pb2.ProgramGraph):
  annotator = liveness.LivenessAnnotator(graph)
  g = annotator.g

  annotator.Annotate(g, 0)
  assert g.graph["data_flow_positive_node_count"] == 2
  assert g.graph["data_flow_steps"] == 4

  # Features:
  assert g.nodes[0]["x"] == [-1, 1]
  assert g.nodes[1]["x"] == [-1, 0]
  assert g.nodes[2]["x"] == [-1, 0]
  assert g.nodes[3]["x"] == [-1, 0]

  assert g.nodes[4]["x"] == [-1, 0]
  assert g.nodes[5]["x"] == [-1, 0]

  # Labels:
  assert g.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[3]["y"] == liveness.NOT_LIVE_OUT

  assert g.nodes[4]["y"] == liveness.LIVE_OUT
  assert g.nodes[5]["y"] == liveness.LIVE_OUT


def test_AnnotateDominatorTree_graph_B(graph: programl_pb2.ProgramGraph):
  annotator = liveness.LivenessAnnotator(graph)
  g = annotator.g

  annotator.Annotate(g, 1)
  assert g.graph["data_flow_positive_node_count"] == 1
  assert g.graph["data_flow_steps"] == 4

  # Features:
  assert g.nodes[0]["x"] == [-1, 0]
  assert g.nodes[1]["x"] == [-1, 1]
  assert g.nodes[2]["x"] == [-1, 0]
  assert g.nodes[3]["x"] == [-1, 0]

  assert g.nodes[4]["x"] == [-1, 0]
  assert g.nodes[5]["x"] == [-1, 0]

  # Labels:
  assert g.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[3]["y"] == liveness.NOT_LIVE_OUT

  assert g.nodes[4]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[5]["y"] == liveness.LIVE_OUT


def test_AnnotateDominatorTree_graph_C(graph: programl_pb2.ProgramGraph):
  annotator = liveness.LivenessAnnotator(graph)
  g = annotator.g

  annotator.Annotate(g, 2)
  assert g.graph["data_flow_positive_node_count"] == 1
  assert g.graph["data_flow_steps"] == 4

  # Features:
  assert g.nodes[0]["x"] == [-1, 0]
  assert g.nodes[1]["x"] == [-1, 0]
  assert g.nodes[2]["x"] == [-1, 1]
  assert g.nodes[3]["x"] == [-1, 0]

  assert g.nodes[4]["x"] == [-1, 0]
  assert g.nodes[5]["x"] == [-1, 0]

  # Labels:
  assert g.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[3]["y"] == liveness.NOT_LIVE_OUT

  assert g.nodes[4]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[5]["y"] == liveness.LIVE_OUT


def test_AnnotateDominatorTree_graph_D(graph: programl_pb2.ProgramGraph):
  annotator = liveness.LivenessAnnotator(graph)
  g = annotator.g

  annotator.Annotate(g, 3)
  assert g.graph["data_flow_positive_node_count"] == 0
  assert g.graph["data_flow_steps"] == 4

  # Features:
  assert g.nodes[0]["x"] == [-1, 0]
  assert g.nodes[1]["x"] == [-1, 0]
  assert g.nodes[2]["x"] == [-1, 0]
  assert g.nodes[3]["x"] == [-1, 1]

  assert g.nodes[4]["x"] == [-1, 0]
  assert g.nodes[5]["x"] == [-1, 0]

  # Labels:
  assert g.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[3]["y"] == liveness.NOT_LIVE_OUT

  assert g.nodes[4]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[5]["y"] == liveness.NOT_LIVE_OUT


###############################################################################
# Graph test.
###############################################################################


@test.Fixture(scope="function")
def while_loop() -> programl_pb2.ProgramGraph:
  """Test fixture which returns a simple "while loop" graph."""
  #          (v1)
  #            |
  #            V
  #    +------[A]<--------+
  #    |       |          |
  #    |       V          |
  #    |      [Ba]----+   |
  #    |       |      |   |
  #    |       |      V   |
  #    |       |     (v3) |
  #    |       V      |   |
  #    |    +-[Bb]<---+   |
  #    |    |  |          |
  #    |    |  +----------+
  #    |  (v2)
  #    V    |
  #   [C]<--+
  builder = programl.GraphBuilder()
  a = builder.AddNode(type=programl_pb2.Node.STATEMENT, x=[-1])  # Loop header
  ba = builder.AddNode(type=programl_pb2.Node.STATEMENT, x=[-1])  # Loop body
  bb = builder.AddNode(type=programl_pb2.Node.STATEMENT, x=[-1])  # Loop body
  c = builder.AddNode(type=programl_pb2.Node.STATEMENT, x=[-1])  # Loop exit

  # Control flow:
  builder.AddEdge(a, ba, flow=programl_pb2.Edge.CONTROL)
  builder.AddEdge(ba, bb, flow=programl_pb2.Edge.CONTROL)
  builder.AddEdge(bb, c, flow=programl_pb2.Edge.CONTROL)
  builder.AddEdge(a, c, flow=programl_pb2.Edge.CONTROL)

  # Data flow:
  v1 = builder.AddNode(
    type=programl_pb2.Node.IDENTIFIER, x=[-1]
  )  # Loop induction variable
  v2 = builder.AddNode(
    type=programl_pb2.Node.IDENTIFIER, x=[-1]
  )  # Computed result
  v3 = builder.AddNode(
    type=programl_pb2.Node.IDENTIFIER, x=[-1]
  )  # Intermediate value

  builder.AddEdge(v1, a, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(ba, v3, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(v3, bb, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(bb, v2, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(v2, c, flow=programl_pb2.Edge.DATA)

  return builder.proto


def test_AnnotateDominatorTree_while_loop_A(
  while_loop: programl_pb2.ProgramGraph,
):
  annotator = liveness.LivenessAnnotator(while_loop)
  g = annotator.g

  annotator.Annotate(g, 0)
  assert g.graph["data_flow_positive_node_count"] == 1
  assert g.graph["data_flow_steps"] == 5

  # Features:
  assert g.nodes[0]["x"] == [-1, 1]
  assert g.nodes[1]["x"] == [-1, 0]
  assert g.nodes[2]["x"] == [-1, 0]
  assert g.nodes[3]["x"] == [-1, 0]

  assert g.nodes[4]["x"] == [-1, 0]
  assert g.nodes[5]["x"] == [-1, 0]
  assert g.nodes[6]["x"] == [-1, 0]

  # Labels:
  assert g.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[3]["y"] == liveness.NOT_LIVE_OUT

  assert g.nodes[4]["y"] == liveness.NOT_LIVE_OUT  # Loop induction variable
  assert g.nodes[5]["y"] == liveness.LIVE_OUT  # Computed result
  assert g.nodes[6]["y"] == liveness.NOT_LIVE_OUT  # Intermediate value


def test_AnnotateDominatorTree_while_loop_Ba(
  while_loop: programl_pb2.ProgramGraph,
):
  annotator = liveness.LivenessAnnotator(while_loop)
  g = annotator.g

  annotator.Annotate(g, 1)
  assert g.graph["data_flow_positive_node_count"] == 1
  assert g.graph["data_flow_steps"] == 5

  # Features:
  assert g.nodes[0]["x"] == [-1, 0]
  assert g.nodes[1]["x"] == [-1, 1]
  assert g.nodes[2]["x"] == [-1, 0]
  assert g.nodes[3]["x"] == [-1, 0]

  assert g.nodes[4]["x"] == [-1, 0]
  assert g.nodes[5]["x"] == [-1, 0]
  assert g.nodes[6]["x"] == [-1, 0]

  # Labels:
  assert g.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[2]["y"] == liveness.NOT_LIVE_OUT

  assert g.nodes[4]["y"] == liveness.NOT_LIVE_OUT  # Loop induction variable
  assert g.nodes[5]["y"] == liveness.NOT_LIVE_OUT  # Computed result
  assert g.nodes[6]["y"] == liveness.LIVE_OUT  # Intermediate value


def test_AnnotateDominatorTree_while_loop_Bb(
  while_loop: programl_pb2.ProgramGraph,
):
  annotator = liveness.LivenessAnnotator(while_loop)
  g = annotator.g

  annotator.Annotate(g, 2)
  assert g.graph["data_flow_positive_node_count"] == 1
  assert g.graph["data_flow_steps"] == 5

  # Features:
  assert g.nodes[0]["x"] == [-1, 0]
  assert g.nodes[1]["x"] == [-1, 0]
  assert g.nodes[2]["x"] == [-1, 1]
  assert g.nodes[3]["x"] == [-1, 0]

  assert g.nodes[4]["x"] == [-1, 0]
  assert g.nodes[5]["x"] == [-1, 0]
  assert g.nodes[6]["x"] == [-1, 0]

  # Labels:
  assert g.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[2]["y"] == liveness.NOT_LIVE_OUT

  assert g.nodes[4]["y"] == liveness.NOT_LIVE_OUT  # Loop induction variable
  assert g.nodes[5]["y"] == liveness.LIVE_OUT  # Computed result
  assert g.nodes[6]["y"] == liveness.NOT_LIVE_OUT  # Intermediate value


def test_AnnotateDominatorTree_while_loop_C(
  while_loop: programl_pb2.ProgramGraph,
):
  annotator = liveness.LivenessAnnotator(while_loop)
  g = annotator.g

  annotator.Annotate(g, 3)
  assert g.graph["data_flow_positive_node_count"] == 0
  assert g.graph["data_flow_steps"] == 5

  # Features:
  assert g.nodes[0]["x"] == [-1, 0]
  assert g.nodes[1]["x"] == [-1, 0]
  assert g.nodes[2]["x"] == [-1, 0]
  assert g.nodes[3]["x"] == [-1, 1]

  assert g.nodes[4]["x"] == [-1, 0]
  assert g.nodes[5]["x"] == [-1, 0]
  assert g.nodes[6]["x"] == [-1, 0]

  # Labels:
  assert g.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert g.nodes[2]["y"] == liveness.NOT_LIVE_OUT

  assert g.nodes[4]["y"] == liveness.NOT_LIVE_OUT  # Loop induction variable
  assert g.nodes[5]["y"] == liveness.NOT_LIVE_OUT  # Computed result
  assert g.nodes[6]["y"] == liveness.NOT_LIVE_OUT  # Intermediate value


###############################################################################
# Tests.
###############################################################################


def test_AnnotateLiveness_exit_block_is_removed(
  wiki: programl_pb2.ProgramGraph,
):
  n = len(wiki.node)
  annotator = liveness.LivenessAnnotator(wiki)
  annotator.Annotate(annotator.g, 5)
  assert n == annotator.g.number_of_nodes()


def test_MakeAnnotated_real_graphs(real_graph: programl_pb2.ProgramGraph,):
  """Opaque black-box test of annotator."""
  annotator = liveness.LivenessAnnotator(real_graph)
  annotated = annotator.MakeAnnotated(10)
  assert len(annotated.graphs) <= 10
  # Assume all graphs produce annotations.
  for graph in annotated.graphs:
    assert graph.graph["data_flow_steps"] >= 1


if __name__ == "__main__":
  test.Main()
