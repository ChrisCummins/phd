"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow/datadep:data_dependence."""
from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from deeplearning.ml4pl.graphs.labelled.dataflow.datadep import data_dependence
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import test

FLAGS = test.FLAGS


###############################################################################
# Fixtures.
###############################################################################


@test.Fixture(
  scope="session", params=list(random_programl_generator.EnumerateTestSet()),
)
def real_proto(request) -> programl_pb2.ProgramGraph:
  """A test fixture which yields one of 100 "real" graphs."""
  return request.param


###############################################################################
# Tests.
###############################################################################


def test_Annotate():
  builder = programl.GraphBuilder()
  A = builder.AddNode(x=[-1])
  B = builder.AddNode(x=[-1])
  C = builder.AddNode(x=[-1])
  D = builder.AddNode(x=[-1])
  E = builder.AddNode(x=[-1])

  builder.AddEdge(A, B)
  builder.AddEdge(A, C)
  builder.AddEdge(B, D)
  builder.AddEdge(C, D)
  builder.AddEdge(A, E)

  builder.AddEdge(A, B, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(A, C, flow=programl_pb2.Edge.DATA)
  builder.AddEdge(C, D, flow=programl_pb2.Edge.DATA)

  g = builder.g

  annotator = data_dependence.DataDependencyAnnotator(builder.proto)

  annotator.Annotate(g, root_node=D)
  assert g.graph["data_flow_positive_node_count"] == 3
  assert g.graph["data_flow_steps"] == 3

  # Features
  assert g.nodes[A]["x"] == [-1, data_flow_graphs.ROOT_NODE_NO]
  assert g.nodes[B]["x"] == [-1, data_flow_graphs.ROOT_NODE_NO]
  assert g.nodes[C]["x"] == [-1, data_flow_graphs.ROOT_NODE_NO]
  assert g.nodes[D]["x"] == [-1, data_flow_graphs.ROOT_NODE_YES]
  assert g.nodes[E]["x"] == [-1, data_flow_graphs.ROOT_NODE_NO]

  # Labels
  assert g.nodes[A]["y"] == data_dependence.DEPENDENCY
  assert g.nodes[B]["y"] == data_dependence.NOT_DEPENDENCY
  assert g.nodes[C]["y"] == data_dependence.DEPENDENCY
  assert g.nodes[D]["y"] == data_dependence.DEPENDENCY
  assert g.nodes[E]["y"] == data_dependence.NOT_DEPENDENCY


def test_MakeAnnotated_real_protos(real_proto: programl_pb2.ProgramGraph,):
  """Opaque black-box test of annotator."""
  annotator = data_dependence.DataDependencyAnnotator(real_proto)
  annotated = annotator.MakeAnnotated(10)
  assert len(annotated.graphs) <= 10


if __name__ == "__main__":
  test.Main()
