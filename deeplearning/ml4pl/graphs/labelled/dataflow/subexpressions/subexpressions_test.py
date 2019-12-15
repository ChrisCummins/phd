"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow/subexpressions."""
import networkx as nx

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow.subexpressions import (
  subexpressions,
)
from deeplearning.ml4pl.testing import random_networkx_generator
from deeplearning.ml4pl.testing import random_programl_generator
from labm8.py import test

FLAGS = test.FLAGS


@test.Fixture(
  scope="session",
  params=list(random_programl_generator.EnumerateProtoTestSet()),
)
def real_proto(request) -> programl_pb2.ProgramGraph:
  """A test fixture which yields one of 100 "real" graphs."""
  return request.param


@test.Fixture(
  scope="session",
  params=list(random_networkx_generator.EnumerateGraphTestSet()),
)
def real_graph(request) -> nx.MultiDiGraph:
  """A test fixture which yields one of 100 "real" graphs."""
  return request.param


@test.Fixture(scope="function")
def wiki() -> programl_pb2.ProgramGraph:
  # a = b * c + g;
  # d = b * c * e;
  builder = programl.GraphBuilder()
  a1 = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])  # 0
  a2 = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])  # 1
  b = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])  # 2
  c = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])  # 3
  d1 = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])  # 4
  d2 = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])  # 5
  e = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])  # 6
  g = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])  # 7

  s0 = builder.AddNode(  # 8
    type=programl_pb2.Node.STATEMENT,
    preprocessed_text="<ID> = mul <ID> <ID>",
    text="%a1 = div %b %c",
    x=[-1],
  )
  s1 = builder.AddNode(  # 9
    type=programl_pb2.Node.STATEMENT,
    preprocessed_text="<ID> = mul <ID> <ID>",
    text="%d1 = div %b %c",
    x=[-1],
  )
  s2 = builder.AddNode(  # 10
    type=programl_pb2.Node.STATEMENT,
    preprocessed_text="<ID> = add <ID> <ID>",
    text="%a2 = add %a1 %g",
    x=[-1],
  )
  s3 = builder.AddNode(  # 11
    type=programl_pb2.Node.STATEMENT,
    preprocessed_text="<ID> = mul <ID> <ID>",
    text="%d2 = mul %d1 %e",
    x=[-1],
  )

  builder.AddEdge(s0, a1, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(b, s0, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(c, s0, flow=programl_pb2.Edge.DATA, position=1)
  builder.AddEdge(s1, d1, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(b, s1, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(c, s1, flow=programl_pb2.Edge.DATA, position=1)

  builder.AddEdge(a1, s2, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(g, s2, flow=programl_pb2.Edge.DATA, position=1)
  builder.AddEdge(s2, a2, flow=programl_pb2.Edge.DATA, position=0)

  builder.AddEdge(d1, s3, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(e, s3, flow=programl_pb2.Edge.DATA, position=1)
  builder.AddEdge(s3, d2, flow=programl_pb2.Edge.DATA, position=0)
  return builder.proto


@test.Fixture(scope="function")
def wiki_without_subexpressions() -> programl_pb2.ProgramGraph:
  """Same as the wiki graph, but the order of the operands for the two
  multiplications has been reversed so that they are no longer common, and
  the operands is not commutative.
  """
  # a = b / c + g;
  # d = c / b * e;
  builder = programl.GraphBuilder()
  a1 = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  a2 = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  b = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  c = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  d1 = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  d2 = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  e = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  g = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])

  s0 = builder.AddNode(
    type=programl_pb2.Node.STATEMENT,
    preprocessed_text="<ID> = sdiv <ID> <ID>",
    text="%a1 = sdiv %b %c",
    x=[-1],
  )
  s1 = builder.AddNode(
    type=programl_pb2.Node.STATEMENT,
    preprocessed_text="<ID> = sdiv <ID> <ID>",
    text="%d1 = sdiv %c %b",
    x=[-1],
  )
  s2 = builder.AddNode(
    type=programl_pb2.Node.STATEMENT,
    preprocessed_text="<ID> = add <ID> <ID>",
    text="%a2 = add %a1 %g",
    x=[-1],
  )
  s3 = builder.AddNode(
    type=programl_pb2.Node.STATEMENT,
    preprocessed_text="<ID> = mul <ID> <ID>",
    text="%d2 = mul %d1 %e",
    x=[-1],
  )

  builder.AddEdge(s0, a1, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(b, s0, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(c, s0, flow=programl_pb2.Edge.DATA, position=1)
  builder.AddEdge(s1, d1, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(b, s1, flow=programl_pb2.Edge.DATA, position=1)
  builder.AddEdge(c, s1, flow=programl_pb2.Edge.DATA, position=0)

  builder.AddEdge(a1, s2, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(g, s2, flow=programl_pb2.Edge.DATA, position=1)
  builder.AddEdge(s2, a2, flow=programl_pb2.Edge.DATA, position=0)

  builder.AddEdge(d1, s3, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(e, s3, flow=programl_pb2.Edge.DATA, position=1)
  builder.AddEdge(s3, d2, flow=programl_pb2.Edge.DATA, position=0)
  return builder.proto


@test.Fixture(scope="function")
def wiki_with_commutativity() -> programl_pb2.ProgramGraph:
  """Same as the wiki graph, but the order of the operands has been reversed
  and the statement is commutative.
  """
  # a = b * c + g;
  # d = c * b * e;
  builder = programl.GraphBuilder()
  a1 = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  a2 = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  b = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  c = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  d1 = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  d2 = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  e = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])
  g = builder.AddNode(type=programl_pb2.Node.IDENTIFIER, x=[-1])

  s0 = builder.AddNode(
    type=programl_pb2.Node.STATEMENT,
    preprocessed_text="<ID> = mul <ID> <ID>",
    text="%a1 = mul %b %c",
    x=[-1],
  )
  s1 = builder.AddNode(
    type=programl_pb2.Node.STATEMENT,
    preprocessed_text="<ID> = mul <ID> <ID>",
    text="%d1 = mul %c %b",
    x=[-1],
  )
  s2 = builder.AddNode(
    type=programl_pb2.Node.STATEMENT,
    preprocessed_text="<ID> = add <ID> <ID>",
    text="%a2 = add %a1 %g",
    x=[-1],
  )
  s3 = builder.AddNode(
    type=programl_pb2.Node.STATEMENT,
    preprocessed_text="<ID> = mul <ID> <ID>",
    text="%d2 = mul %d1 %e",
    x=[-1],
  )

  builder.AddEdge(s0, a1, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(b, s0, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(c, s0, flow=programl_pb2.Edge.DATA, position=1)
  builder.AddEdge(s1, d1, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(b, s1, flow=programl_pb2.Edge.DATA, position=1)
  builder.AddEdge(c, s1, flow=programl_pb2.Edge.DATA, position=0)

  builder.AddEdge(a1, s2, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(g, s2, flow=programl_pb2.Edge.DATA, position=1)
  builder.AddEdge(s2, a2, flow=programl_pb2.Edge.DATA, position=0)

  builder.AddEdge(d1, s3, flow=programl_pb2.Edge.DATA, position=0)
  builder.AddEdge(e, s3, flow=programl_pb2.Edge.DATA, position=1)
  builder.AddEdge(s3, d2, flow=programl_pb2.Edge.DATA, position=0)
  return builder.proto


def test_GetExpressionSets_wiki(wiki: programl_pb2.ProgramGraph):
  expressions = subexpressions.GetExpressionSets(
    programl.ProgramGraphToNetworkX(wiki)
  )
  assert sorted(expressions) == [
    {8, 9},
    {10},
    {11},
  ]


def test_GetExpressionSets_wiki_without_subexpressions(
  wiki_without_subexpressions: programl_pb2.ProgramGraph,
):
  expressions = subexpressions.GetExpressionSets(
    programl.ProgramGraphToNetworkX(wiki_without_subexpressions)
  )
  assert sorted(expressions) == [
    {8},
    {9},
    {10},
    {11},
  ]


def test_GetExpressionSets_wiki_with_commutativity(
  wiki: programl_pb2.ProgramGraph,
  wiki_with_commutativity: programl_pb2.ProgramGraph,
):
  """Test that commutative operands yield the same expression sets."""
  assert subexpressions.GetExpressionSets(
    programl.ProgramGraphToNetworkX(wiki)
  ) == subexpressions.GetExpressionSets(
    programl.ProgramGraphToNetworkX(wiki_with_commutativity)
  )


def test_MakeSubexpressionsGraphs_wiki(wiki: programl_pb2.ProgramGraph):
  """Test the labels generated by a graph with a common subexpression."""
  annotator = subexpressions.CommonSubexpressionAnnotator(wiki)
  annotated = annotator.MakeAnnotated()
  assert len(annotated.graphs) == 2

  g = annotated.graphs[0]

  # Features
  assert g.nodes[0]["x"] == [-1, 0]  # a1
  assert g.nodes[1]["x"] == [-1, 0]  # a2
  assert g.nodes[2]["x"] == [-1, 0]  # b
  assert g.nodes[3]["x"] == [-1, 0]  # c
  assert g.nodes[4]["x"] == [-1, 0]  # d1
  assert g.nodes[5]["x"] == [-1, 0]  # d2
  assert g.nodes[6]["x"] == [-1, 0]  # e
  assert g.nodes[7]["x"] == [-1, 0]  # g

  assert g.nodes[8]["x"] != g.nodes[9]["x"]  # s0
  assert g.nodes[10]["x"] == [-1, 0]  # s1
  assert g.nodes[11]["x"] == [-1, 0]  # s2

  # Labels
  assert g.nodes[0]["y"] == subexpressions.NOT_COMMON_SUBEXPRESSION
  assert g.nodes[1]["y"] == subexpressions.NOT_COMMON_SUBEXPRESSION
  assert g.nodes[2]["y"] == subexpressions.NOT_COMMON_SUBEXPRESSION
  assert g.nodes[3]["y"] == subexpressions.NOT_COMMON_SUBEXPRESSION
  assert g.nodes[4]["y"] == subexpressions.NOT_COMMON_SUBEXPRESSION
  assert g.nodes[5]["y"] == subexpressions.NOT_COMMON_SUBEXPRESSION
  assert g.nodes[6]["y"] == subexpressions.NOT_COMMON_SUBEXPRESSION
  assert g.nodes[7]["y"] == subexpressions.NOT_COMMON_SUBEXPRESSION

  assert g.nodes[8]["y"] == subexpressions.COMMON_SUBEXPRESSION
  assert g.nodes[9]["y"] == subexpressions.COMMON_SUBEXPRESSION
  assert g.nodes[10]["y"] == subexpressions.NOT_COMMON_SUBEXPRESSION
  assert g.nodes[11]["y"] == subexpressions.NOT_COMMON_SUBEXPRESSION


def test_GetExpressionSets_commutative_graph_labels(
  wiki: programl_pb2.ProgramGraph,
  wiki_with_commutativity: programl_pb2.ProgramGraph,
):
  """Test that commutative ops produce the same labels."""
  a = subexpressions.CommonSubexpressionAnnotator(wiki)
  graphs_a = a.MakeAnnotated().graphs
  b = subexpressions.CommonSubexpressionAnnotator(wiki_with_commutativity)
  graphs_b = b.MakeAnnotated().graphs
  assert len(graphs_a) == len(graphs_b) == 2
  a, b = graphs_a[0], graphs_b[0]
  for node in a.nodes():
    # Note we can't test for equality of 'x' because the root node is chosen
    # randomly.
    assert a.nodes[node]["y"] == b.nodes[node]["y"]


def test_GetExpressionSets_statement_count(real_graph: nx.MultiDiGraph):
  """Check that every statement node appears in the expression sets."""
  expression_sets = subexpressions.GetExpressionSets(real_graph)
  # Count the number of statements with at least one operand.
  statements_with_operands_count = len(
    [
      n
      for n, type_ in real_graph.nodes(data="type")
      if type_ == programl_pb2.Node.STATEMENT
      and any(
        _
        for _, _, flow in real_graph.in_edges(n, data="flow")
        if flow == programl_pb2.Edge.DATA
      )
    ]
  )
  flattened_expression_sets = set().union(*expression_sets)
  assert statements_with_operands_count == len(flattened_expression_sets)


def test_GetExpressionSets_does_not_include_duplicates(
  real_graph: nx.MultiDiGraph,
):
  """Check that every node appears in only one expression set."""
  expression_sets = subexpressions.GetExpressionSets(real_graph)
  flattened_expression_sets = set().union(*expression_sets)
  assert len(flattened_expression_sets) == sum(len(s) for s in expression_sets)


def test_MakeSubexpressionsGraphs_wiki_without_subexpressions(
  wiki_without_subexpressions: programl_pb2.ProgramGraph,
):
  """Test that graph without common subexpressions yields no outputs."""
  annotator = subexpressions.CommonSubexpressionAnnotator(
    wiki_without_subexpressions
  )
  graphs = annotator.MakeAnnotated().graphs
  assert len(graphs) == 0


def test_MakeAnnotated_real_protos(real_proto: programl_pb2.ProgramGraph,):
  """Opaque black-box test of reachability annotator."""
  annotator = subexpressions.CommonSubexpressionAnnotator(real_proto)
  annotated = annotator.MakeAnnotated(10)
  assert len(annotated.graphs) <= 10


if __name__ == "__main__":
  test.Main()
