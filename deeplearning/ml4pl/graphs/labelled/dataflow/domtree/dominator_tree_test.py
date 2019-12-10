"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow/domtree:dominator_tree."""
import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow.domtree import dominator_tree
from labm8.py import test

FLAGS = test.FLAGS

###############################################################################
# Fixtures.
###############################################################################


@test.Fixture(scope="session")
def graph() -> nx.MultiDiGraph():
  builder = programl.GraphBuilder()
  a = builder.AddNode(x=[-1])
  b = builder.AddNode(x=[-1])
  c = builder.AddNode(x=[-1])
  d = builder.AddNode(x=[-1])
  e = builder.AddNode(x=[-1])
  v1 = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  builder.AddEdge(a, b)
  builder.AddEdge(a, c)
  builder.AddEdge(b, d)
  builder.AddEdge(c, d)
  builder.AddEdge(v1, c)
  builder.AddEdge(a, c)
  builder.AddEdge(e, b)
  return builder.g


@test.Fixture(scope="function")
def annotator() -> dominator_tree.DominatorTreeAnnotator:
  return dominator_tree.DominatorTreeAnnotator()


###############################################################################
# Tests.
###############################################################################


def test_AnnotateDominatorTree(
  graph: nx.MultiDiGraph, annotator: dominator_tree.DominatorTreeAnnotator
):
  annotated = annotator.Annotate(graph, root_node=0)

  assert annotated.data_flow_positive_node_count == 2
  # TODO(cec): What is the correct value here?
  assert annotated.data_flow_steps in {2, 3}

  # Features
  assert annotated.node[0].x == [-1, 1]
  assert annotated.node[1].x == [-1, 0]
  assert annotated.node[2].x == [-1, 0]
  assert annotated.node[3].x == [-1, 0]
  assert annotated.node[4].x == [-1, 0]
  assert annotated.node[5].x == [-1, 0]

  # Labels
  assert annotated.node[0].y == dominator_tree.DOMINATED
  assert annotated.node[1].y == dominator_tree.NOT_DOMINATED
  assert annotated.node[2].y == dominator_tree.DOMINATED
  assert annotated.node[3].y == dominator_tree.NOT_DOMINATED
  assert annotated.node[4].y == dominator_tree.NOT_DOMINATED
  assert annotated.node[5].y == dominator_tree.NOT_DOMINATED


if __name__ == "__main__":
  test.Main()
