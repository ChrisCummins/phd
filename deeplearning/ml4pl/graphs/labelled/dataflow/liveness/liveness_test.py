"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow/liveness."""
import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow.liveness import liveness
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def annotator() -> liveness.LivenessAnnotator:
  """A test fixture which yield an annotator."""
  return liveness.LivenessAnnotator()


@test.Fixture(scope="function")
def wiki() -> nx.MultiDiGraph:
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
  return builder.g


def test_AnnotateLiveness_exit_block_is_removed(
  wiki: nx.MultiDiGraph, annotator: liveness.LivenessAnnotator
):
  n = wiki.number_of_nodes()
  assert n == annotator.Annotate(wiki, 5).number_of_nodes()


def test_AnnotateLiveness_wiki_b1(
  wiki: nx.MultiDiGraph, annotator: liveness.LivenessAnnotator
):
  """Test liveness annotations from block b1."""
  annotated = annotator.Annotate(wiki, 5)
  assert annotated.graph["data_flow_positive_node_count"] == 3
  assert annotated.graph["data_flow_steps"] == 5

  # Features:
  assert wiki.nodes[5]["x"] == [-1, 1]
  assert wiki.nodes[6]["x"] == [-1, 0]
  assert wiki.nodes[7]["x"] == [-1, 0]
  assert wiki.nodes[8]["x"] == [-1, 0]

  assert wiki.nodes[0]["x"] == [-1, 0]
  assert wiki.nodes[1]["x"] == [-1, 0]
  assert wiki.nodes[2]["x"] == [-1, 0]
  assert wiki.nodes[3]["x"] == [-1, 0]

  # Labels:
  assert wiki.nodes[5]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[6]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[7]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[8]["y"] == liveness.NOT_LIVE_OUT

  assert wiki.nodes[0]["y"] == liveness.LIVE_OUT
  assert wiki.nodes[1]["y"] == liveness.LIVE_OUT
  assert wiki.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[3]["y"] == liveness.LIVE_OUT


def test_AnnotateLiveness_wiki_b2(
  wiki: nx.MultiDiGraph, annotator: liveness.LivenessAnnotator
):
  """Test liveness annotations from block b2."""
  annotated = annotator.Annotate(wiki, 6)
  assert annotated.graph["data_flow_positive_node_count"] == 2
  assert annotated.graph["data_flow_steps"] == 5

  # Features:
  assert wiki.nodes[5]["x"] == [-1, 0]
  assert wiki.nodes[6]["x"] == [-1, 1]
  assert wiki.nodes[7]["x"] == [-1, 0]
  assert wiki.nodes[8]["x"] == [-1, 0]

  assert wiki.nodes[0]["x"] == [-1, 0]
  assert wiki.nodes[1]["x"] == [-1, 0]
  assert wiki.nodes[2]["x"] == [-1, 0]
  assert wiki.nodes[3]["x"] == [-1, 0]

  # Labels:
  assert wiki.nodes[5]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[6]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[7]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[8]["y"] == liveness.NOT_LIVE_OUT

  assert wiki.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[1]["y"] == liveness.LIVE_OUT
  assert wiki.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[3]["y"] == liveness.LIVE_OUT


def test_AnnotateLiveness_wiki_b3a(
  wiki: nx.MultiDiGraph, annotator: liveness.LivenessAnnotator
):
  """Test liveness annotations from block b3a."""
  annotated = annotator.Annotate(wiki, 7)
  assert annotated.graph["data_flow_positive_node_count"] == 3
  assert annotated.graph["data_flow_steps"] == 5

  # Features:
  assert wiki.nodes[5]["x"] == [-1, 0]
  assert wiki.nodes[6]["x"] == [-1, 0]
  assert wiki.nodes[7]["x"] == [-1, 1]
  assert wiki.nodes[8]["x"] == [-1, 0]

  assert wiki.nodes[0]["x"] == [-1, 0]
  assert wiki.nodes[1]["x"] == [-1, 0]
  assert wiki.nodes[2]["x"] == [-1, 0]
  assert wiki.nodes[3]["x"] == [-1, 0]

  # Labels:
  assert wiki.nodes[5]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[6]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[7]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[8]["y"] == liveness.NOT_LIVE_OUT

  assert wiki.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[1]["y"] == liveness.LIVE_OUT
  assert wiki.nodes[2]["y"] == liveness.LIVE_OUT
  assert wiki.nodes[3]["y"] == liveness.LIVE_OUT


def test_AnnotateLiveness_wiki_b3b(
  wiki: nx.MultiDiGraph, annotator: liveness.LivenessAnnotator
):
  """Test liveness annotations from block b3b."""
  annotated = annotator.Annotate(wiki, 8)
  assert annotated.graph["data_flow_positive_node_count"] == 0
  assert annotated.graph["data_flow_steps"] == 5

  # Features:
  assert wiki.nodes[5]["x"] == [-1, 0]
  assert wiki.nodes[6]["x"] == [-1, 0]
  assert wiki.nodes[7]["x"] == [-1, 0]
  assert wiki.nodes[8]["x"] == [-1, 1]

  assert wiki.nodes[0]["x"] == [-1, 0]
  assert wiki.nodes[1]["x"] == [-1, 0]
  assert wiki.nodes[2]["x"] == [-1, 0]
  assert wiki.nodes[3]["x"] == [-1, 0]

  # Labels:
  assert wiki.nodes[5]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[6]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[7]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[8]["y"] == liveness.NOT_LIVE_OUT

  assert wiki.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert wiki.nodes[3]["y"] == liveness.NOT_LIVE_OUT


@test.Fixture(scope="function")
def graph() -> nx.MultiDiGraph:
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

  return builder.g


def test_AnnotateDominatorTree_graph_A(
  graph: nx.MultiDiGraph, annotator: liveness.LivenessAnnotator
):
  annotated = annotator.Annotate(graph, 0)
  assert annotated.graph["data_flow_positive_node_count"] == 2
  assert annotated.graph["data_flow_steps"] == 4

  # Features:
  assert graph.nodes[0]["x"] == [-1, 1]
  assert graph.nodes[1]["x"] == [-1, 0]
  assert graph.nodes[2]["x"] == [-1, 0]
  assert graph.nodes[3]["x"] == [-1, 0]

  assert graph.nodes[4]["x"] == [-1, 0]
  assert graph.nodes[5]["x"] == [-1, 0]

  # Labels:
  assert graph.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert graph.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert graph.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert graph.nodes[3]["y"] == liveness.NOT_LIVE_OUT

  assert graph.nodes[4]["y"] == liveness.LIVE_OUT
  assert graph.nodes[5]["y"] == liveness.LIVE_OUT


def test_AnnotateDominatorTree_graph_B(
  graph: nx.MultiDiGraph, annotator: liveness.LivenessAnnotator
):
  annotated = annotator.Annotate(graph, 1)
  assert annotated.graph["data_flow_positive_node_count"] == 1
  assert annotated.graph["data_flow_steps"] == 4

  # Features:
  assert graph.nodes[0]["x"] == [-1, 0]
  assert graph.nodes[1]["x"] == [-1, 1]
  assert graph.nodes[2]["x"] == [-1, 0]
  assert graph.nodes[3]["x"] == [-1, 0]

  assert graph.nodes[4]["x"] == [-1, 0]
  assert graph.nodes[5]["x"] == [-1, 0]

  # Labels:
  assert graph.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert graph.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert graph.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert graph.nodes[3]["y"] == liveness.NOT_LIVE_OUT

  assert graph.nodes[4]["y"] == liveness.NOT_LIVE_OUT
  assert graph.nodes[5]["y"] == liveness.LIVE_OUT


def test_AnnotateDominatorTree_graph_C(
  graph: nx.MultiDiGraph, annotator: liveness.LivenessAnnotator
):
  annotated = annotator.Annotate(graph, 2)
  assert annotated.graph["data_flow_positive_node_count"] == 1
  assert annotated.graph["data_flow_steps"] == 4

  # Features:
  assert graph.nodes[0]["x"] == [-1, 0]
  assert graph.nodes[1]["x"] == [-1, 0]
  assert graph.nodes[2]["x"] == [-1, 1]
  assert graph.nodes[3]["x"] == [-1, 0]

  assert graph.nodes[4]["x"] == [-1, 0]
  assert graph.nodes[5]["x"] == [-1, 0]

  # Labels:
  assert graph.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert graph.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert graph.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert graph.nodes[3]["y"] == liveness.NOT_LIVE_OUT

  assert graph.nodes[4]["y"] == liveness.NOT_LIVE_OUT
  assert graph.nodes[5]["y"] == liveness.LIVE_OUT


def test_AnnotateDominatorTree_graph_D(
  graph: nx.MultiDiGraph, annotator: liveness.LivenessAnnotator
):
  annotated = annotator.Annotate(graph, 3)
  assert annotated.graph["data_flow_positive_node_count"] == 0
  assert annotated.graph["data_flow_steps"] == 4

  # Features:
  assert graph.nodes[0]["x"] == [-1, 0]
  assert graph.nodes[1]["x"] == [-1, 0]
  assert graph.nodes[2]["x"] == [-1, 0]
  assert graph.nodes[3]["x"] == [-1, 1]

  assert graph.nodes[4]["x"] == [-1, 0]
  assert graph.nodes[5]["x"] == [-1, 0]

  # Labels:
  assert graph.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert graph.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert graph.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert graph.nodes[3]["y"] == liveness.NOT_LIVE_OUT

  assert graph.nodes[4]["y"] == liveness.NOT_LIVE_OUT
  assert graph.nodes[5]["y"] == liveness.NOT_LIVE_OUT


@test.Fixture(scope="function")
def while_loop() -> nx.MultiDiGraph:
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

  return builder.g


def test_AnnotateDominatorTree_while_loop_A(
  while_loop: nx.MultiDiGraph, annotator: liveness.LivenessAnnotator
):
  annotated = annotator.Annotate(while_loop, 0)
  assert annotated.graph["data_flow_positive_node_count"] == 1
  assert annotated.graph["data_flow_steps"] == 5

  # Features:
  assert while_loop.nodes[0]["x"] == [-1, 1]
  assert while_loop.nodes[1]["x"] == [-1, 0]
  assert while_loop.nodes[2]["x"] == [-1, 0]
  assert while_loop.nodes[3]["x"] == [-1, 0]

  assert while_loop.nodes[4]["x"] == [-1, 0]
  assert while_loop.nodes[5]["x"] == [-1, 0]
  assert while_loop.nodes[6]["x"] == [-1, 0]

  # Labels:
  assert while_loop.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert while_loop.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert while_loop.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert while_loop.nodes[3]["y"] == liveness.NOT_LIVE_OUT

  assert (
    while_loop.nodes[4]["y"] == liveness.NOT_LIVE_OUT
  )  # Loop induction variable
  assert while_loop.nodes[5]["y"] == liveness.LIVE_OUT  # Computed result
  assert while_loop.nodes[6]["y"] == liveness.NOT_LIVE_OUT  # Intermediate value


def test_AnnotateDominatorTree_while_loop_Ba(
  while_loop: nx.MultiDiGraph, annotator: liveness.LivenessAnnotator
):
  annotated = annotator.Annotate(while_loop, 1)
  assert annotated.graph["data_flow_positive_node_count"] == 1
  assert annotated.graph["data_flow_steps"] == 5

  # Features:
  assert while_loop.nodes[0]["x"] == [-1, 0]
  assert while_loop.nodes[1]["x"] == [-1, 1]
  assert while_loop.nodes[2]["x"] == [-1, 0]
  assert while_loop.nodes[3]["x"] == [-1, 0]

  assert while_loop.nodes[4]["x"] == [-1, 0]
  assert while_loop.nodes[5]["x"] == [-1, 0]
  assert while_loop.nodes[6]["x"] == [-1, 0]

  # Labels:
  assert while_loop.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert while_loop.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert while_loop.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert while_loop.nodes[2]["y"] == liveness.NOT_LIVE_OUT

  assert (
    while_loop.nodes[4]["y"] == liveness.NOT_LIVE_OUT
  )  # Loop induction variable
  assert while_loop.nodes[5]["y"] == liveness.NOT_LIVE_OUT  # Computed result
  assert while_loop.nodes[6]["y"] == liveness.LIVE_OUT  # Intermediate value


def test_AnnotateDominatorTree_while_loop_Bb(
  while_loop: nx.MultiDiGraph, annotator: liveness.LivenessAnnotator
):
  annotated = annotator.Annotate(while_loop, 2)
  assert annotated.graph["data_flow_positive_node_count"] == 1
  assert annotated.graph["data_flow_steps"] == 5

  # Features:
  assert while_loop.nodes[0]["x"] == [-1, 0]
  assert while_loop.nodes[1]["x"] == [-1, 0]
  assert while_loop.nodes[2]["x"] == [-1, 1]
  assert while_loop.nodes[3]["x"] == [-1, 0]

  assert while_loop.nodes[4]["x"] == [-1, 0]
  assert while_loop.nodes[5]["x"] == [-1, 0]
  assert while_loop.nodes[6]["x"] == [-1, 0]

  # Labels:
  assert while_loop.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert while_loop.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert while_loop.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert while_loop.nodes[2]["y"] == liveness.NOT_LIVE_OUT

  assert (
    while_loop.nodes[4]["y"] == liveness.NOT_LIVE_OUT
  )  # Loop induction variable
  assert while_loop.nodes[5]["y"] == liveness.LIVE_OUT  # Computed result
  assert while_loop.nodes[6]["y"] == liveness.NOT_LIVE_OUT  # Intermediate value


def test_AnnotateDominatorTree_while_loop_C(
  while_loop: nx.MultiDiGraph, annotator: liveness.LivenessAnnotator
):
  annotated = annotator.Annotate(while_loop, 3)
  assert annotated.graph["data_flow_positive_node_count"] == 0
  assert annotated.graph["data_flow_steps"] == 5

  # Features:
  assert while_loop.nodes[0]["x"] == [-1, 0]
  assert while_loop.nodes[1]["x"] == [-1, 0]
  assert while_loop.nodes[2]["x"] == [-1, 0]
  assert while_loop.nodes[3]["x"] == [-1, 1]

  assert while_loop.nodes[4]["x"] == [-1, 0]
  assert while_loop.nodes[5]["x"] == [-1, 0]
  assert while_loop.nodes[6]["x"] == [-1, 0]

  # Labels:
  assert while_loop.nodes[0]["y"] == liveness.NOT_LIVE_OUT
  assert while_loop.nodes[1]["y"] == liveness.NOT_LIVE_OUT
  assert while_loop.nodes[2]["y"] == liveness.NOT_LIVE_OUT
  assert while_loop.nodes[2]["y"] == liveness.NOT_LIVE_OUT

  assert (
    while_loop.nodes[4]["y"] == liveness.NOT_LIVE_OUT
  )  # Loop induction variable
  assert while_loop.nodes[5]["y"] == liveness.NOT_LIVE_OUT  # Computed result
  assert while_loop.nodes[6]["y"] == liveness.NOT_LIVE_OUT  # Intermediate value


if __name__ == "__main__":
  test.Main()
