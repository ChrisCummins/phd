"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow/subexpressions."""
import networkx as nx
import pytest

from deeplearning.ml4pl.graphs.labelled.dataflow.subexpressions import (
  subexpressions,
)
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def wiki() -> nx.MultiDiGraph:
  # a = b * c + g;
  # d = b * c * e;
  g = nx.MultiDiGraph()
  g.add_node("a1", type="identifier", name="a1", x=-1)
  g.add_node("a2", type="identifier", name="a2", x=-1)
  g.add_node("b", type="identifier", name="b", x=-1)
  g.add_node("c", type="identifier", name="c", x=-1)
  g.add_node("d1", type="identifier", name="d1", x=-1)
  g.add_node("d2", type="identifier", name="d2", x=-1)
  g.add_node("e", type="identifier", name="e", x=-1)
  g.add_node("g", type="identifier", name="g", x=-1)

  g.add_node(
    "s0",
    type="statement",
    text="<ID> = mul <ID> <ID>",
    original_text="%a1 = div %b %c",
    x=-1,
  )
  g.add_node(
    "s1",
    type="statement",
    text="<ID> = mul <ID> <ID>",
    original_text="%d1 = div %b %c",
    x=-1,
  )
  g.add_node(
    "s2",
    type="statement",
    text="<ID> = add <ID> <ID>",
    original_text="%a2 = add %a1 %g",
    x=-1,
  )
  g.add_node(
    "s3",
    type="statement",
    text="<ID> = mul <ID> <ID>",
    original_text="%d2 = mul %d1 %e",
    x=-1,
  )

  g.add_edge("s0", "a1", flow="data", position=0)
  g.add_edge("b", "s0", flow="data", position=0)
  g.add_edge("c", "s0", flow="data", position=1)
  g.add_edge("s1", "d1", flow="data", position=0)
  g.add_edge("b", "s1", flow="data", position=0)
  g.add_edge("c", "s1", flow="data", position=1)

  g.add_edge("a1", "s2", flow="data", position=0)
  g.add_edge("g", "s2", flow="data", position=1)
  g.add_edge("s2", "a2", flow="data", position=0)

  g.add_edge("d1", "s3", flow="data", position=0)
  g.add_edge("e", "s3", flow="data", position=1)
  g.add_edge("s3", "d2", flow="data", position=0)
  return g


@test.Fixture(scope="function")
def wiki_without_subexpressions() -> nx.MultiDiGraph:
  """Same as the wiki graph, but the order of the operands for the two
  multiplications has been reversed so that they are no longer common, and
  the operands is not commutative.
  """
  # a = b / c + g;
  # d = c / b * e;
  g = nx.MultiDiGraph()
  g.add_node("a1", type="identifier", name="a1", x=-1)
  g.add_node("a2", type="identifier", name="a2", x=-1)
  g.add_node("b", type="identifier", name="b", x=-1)
  g.add_node("c", type="identifier", name="c", x=-1)
  g.add_node("d1", type="identifier", name="d1", x=-1)
  g.add_node("d2", type="identifier", name="d2", x=-1)
  g.add_node("e", type="identifier", name="e", x=-1)
  g.add_node("g", type="identifier", name="g", x=-1)

  g.add_node(
    "s0",
    type="statement",
    text="<ID> = sdiv <ID> <ID>",
    original_text="%a1 = sdiv %b %c",
    x=-1,
  )
  g.add_node(
    "s1",
    type="statement",
    text="<ID> = sdiv <ID> <ID>",
    original_text="%d1 = sdiv %c %b",
    x=-1,
  )
  g.add_node(
    "s2",
    type="statement",
    text="<ID> = add <ID> <ID>",
    original_text="%a2 = add %a1 %g",
    x=-1,
  )
  g.add_node(
    "s3",
    type="statement",
    text="<ID> = mul <ID> <ID>",
    original_text="%d2 = mul %d1 %e",
    x=-1,
  )

  g.add_edge("s0", "a1", flow="data", position=0)
  g.add_edge("b", "s0", flow="data", position=0)
  g.add_edge("c", "s0", flow="data", position=1)
  g.add_edge("s1", "d1", flow="data", position=0)
  g.add_edge("b", "s1", flow="data", position=1)
  g.add_edge("c", "s1", flow="data", position=0)

  g.add_edge("a1", "s2", flow="data", position=0)
  g.add_edge("g", "s2", flow="data", position=1)
  g.add_edge("s2", "a2", flow="data", position=0)

  g.add_edge("d1", "s3", flow="data", position=0)
  g.add_edge("e", "s3", flow="data", position=1)
  g.add_edge("s3", "d2", flow="data", position=0)
  return g


@test.Fixture(scope="function")
def wiki_with_commutativity() -> nx.MultiDiGraph:
  """Same as the wiki graph, but the order of the operands has been reversed
  and the statement is commutative.
  """
  # a = b * c + g;
  # d = c * b * e;
  g = nx.MultiDiGraph()
  g.add_node("a1", type="identifier", name="a1", x=-1)
  g.add_node("a2", type="identifier", name="a2", x=-1)
  g.add_node("b", type="identifier", name="b", x=-1)
  g.add_node("c", type="identifier", name="c", x=-1)
  g.add_node("d1", type="identifier", name="d1", x=-1)
  g.add_node("d2", type="identifier", name="d2", x=-1)
  g.add_node("e", type="identifier", name="e", x=-1)
  g.add_node("g", type="identifier", name="g", x=-1)

  g.add_node(
    "s0",
    type="statement",
    text="<ID> = mul <ID> <ID>",
    original_text="%a1 = mul %b %c",
    x=-1,
  )
  g.add_node(
    "s1",
    type="statement",
    text="<ID> = mul <ID> <ID>",
    original_text="%d1 = mul %c %b",
    x=-1,
  )
  g.add_node(
    "s2",
    type="statement",
    text="<ID> = add <ID> <ID>",
    original_text="%a2 = add %a1 %g",
    x=-1,
  )
  g.add_node(
    "s3",
    type="statement",
    text="<ID> = mul <ID> <ID>",
    original_text="%d2 = mul %d1 %e",
    x=-1,
  )

  g.add_edge("s0", "a1", flow="data", position=0)
  g.add_edge("b", "s0", flow="data", position=0)
  g.add_edge("c", "s0", flow="data", position=1)
  g.add_edge("s1", "d1", flow="data", position=0)
  g.add_edge("b", "s1", flow="data", position=1)
  g.add_edge("c", "s1", flow="data", position=0)

  g.add_edge("a1", "s2", flow="data", position=0)
  g.add_edge("g", "s2", flow="data", position=1)
  g.add_edge("s2", "a2", flow="data", position=0)

  g.add_edge("d1", "s3", flow="data", position=0)
  g.add_edge("e", "s3", flow="data", position=1)
  g.add_edge("s3", "d2", flow="data", position=0)
  return g


def test_GetExpressionSets_wiki(wiki: nx.MultiDiGraph):
  expressions = subexpressions.GetExpressionSets(wiki)
  assert expressions == {
    ("mul <ID> <ID>", ("b", "c")): ["s0", "s1"],
    ("add <ID> <ID>", ("a1", "g")): ["s2"],
    ("mul <ID> <ID>", ("d1", "e")): ["s3"],
  }


def test_GetExpressionSets_wiki_without_subexpressions(
  wiki_without_subexpressions: nx.MultiDiGraph,
):
  expressions = subexpressions.GetExpressionSets(wiki_without_subexpressions)
  assert expressions == {
    ("sdiv <ID> <ID>", ("b", "c")): ["s0"],
    ("sdiv <ID> <ID>", ("c", "b")): ["s1"],
    ("add <ID> <ID>", ("a1", "g")): ["s2"],
    ("mul <ID> <ID>", ("d1", "e")): ["s3"],
  }


def test_GetExpressionSets_wiki_with_commutativity(
  wiki: nx.MultiDiGraph, wiki_with_commutativity: nx.MultiDiGraph
):
  """Test that commutative operands yield the same expression sets."""
  assert subexpressions.GetExpressionSets(
    wiki
  ) == subexpressions.GetExpressionSets(wiki_with_commutativity)


def test_MakeSubexpressionsGraphs_wiki(wiki: nx.MultiDiGraph):
  """Test the labels generated by a graph with a common subexpression."""
  graphs = list(subexpressions.MakeSubexpressionsGraphs(wiki))
  assert len(graphs) == 1

  g = graphs[0]

  # Features
  assert g.nodes["a1"]["x"] == [-1, 0]
  assert g.nodes["a2"]["x"] == [-1, 0]
  assert g.nodes["b"]["x"] == [-1, 0]
  assert g.nodes["c"]["x"] == [-1, 0]
  assert g.nodes["d1"]["x"] == [-1, 0]
  assert g.nodes["d2"]["x"] == [-1, 0]
  assert g.nodes["e"]["x"] == [-1, 0]
  assert g.nodes["g"]["x"] == [-1, 0]

  assert g.nodes["s0"]["x"] != g.nodes["s1"]["x"]
  assert g.nodes["s2"]["x"] == [-1, 0]
  assert g.nodes["s3"]["x"] == [-1, 0]

  # Labels
  assert g.nodes["a1"]["y"] == False
  assert g.nodes["a2"]["y"] == False
  assert g.nodes["b"]["y"] == False
  assert g.nodes["c"]["y"] == False
  assert g.nodes["d1"]["y"] == False
  assert g.nodes["d2"]["y"] == False
  assert g.nodes["e"]["y"] == False
  assert g.nodes["g"]["y"] == False

  assert g.nodes["s0"]["y"] == True
  assert g.nodes["s1"]["y"] == True
  assert g.nodes["s2"]["y"] == False
  assert g.nodes["s3"]["y"] == False


def test_GetExpressionSets_commutative_graph_labels(
  wiki: nx.MultiDiGraph, wiki_with_commutativity: nx.MultiDiGraph
):
  """Test that commutative ops produce the same labels."""
  graphs_a = list(subexpressions.MakeSubexpressionsGraphs(wiki))
  graphs_b = list(
    subexpressions.MakeSubexpressionsGraphs(wiki_with_commutativity)
  )
  assert len(graphs_a) == len(graphs_b) == 1
  a, b = graphs_a[0], graphs_b[0]
  for node in a.nodes():
    # Note we can't test for equality of 'x' because the root node is chosen
    # randomly.
    assert a.nodes[node]["y"] == b.nodes[node]["y"]


def test_MakeSubexpressionsGraphs_wiki_without_subexpressions(
  wiki_without_subexpressions: nx.MultiDiGraph,
):
  """Test that graph without common subexpressions yields no outputs."""
  graphs = list(
    subexpressions.MakeSubexpressionsGraphs(wiki_without_subexpressions)
  )
  assert len(graphs) == 0


if __name__ == "__main__":
  test.Main()
