"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow/liveness."""
import networkx as nx
import pytest

from deeplearning.ml4pl.graphs.labelled.dataflow.liveness import liveness
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def wiki() -> nx.MultiDiGraph:
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
  g = nx.MultiDiGraph()

  # Variables:
  g.add_node("a", type="identifier", x=-1)
  g.add_node("b", type="identifier", x=-1)
  g.add_node("c", type="identifier", x=-1)
  g.add_node("d", type="identifier", x=-1)
  g.add_node("x", type="identifier", x=-1)

  # b1
  g.add_node("b1", type="statement", x=-1)
  # Defs
  g.add_edge("b1", "a", flow="data")
  g.add_edge("b1", "b", flow="data")
  g.add_edge("b1", "d", flow="data")
  g.add_edge("b1", "x", flow="data")

  # b2
  g.add_node("b2", type="statement", x=-1)
  g.add_edge("b1", "b2", flow="control")
  # Defs
  g.add_edge("b2", "c", flow="data")
  g.add_edge("b2", "d", flow="data")
  # Uses
  g.add_edge("a", "b2", flow="data")
  g.add_edge("b", "b2", flow="data")

  # b3a
  g.add_node("b3a", type="statement", x=-1)
  g.add_edge("b1", "b3a", flow="control")
  g.add_edge("b2", "b3a", flow="control")
  # Defs
  g.add_edge("b3a", "c", flow="data")

  # b3b
  g.add_node("b3b", type="statement", x=-1)
  g.add_edge("b3a", "b3b", flow="control")
  # Uses
  g.add_edge("b", "b3b", flow="data")
  g.add_edge("d", "b3b", flow="data")
  g.add_edge("c", "b3b", flow="data")
  return g


def test_AnnotateLiveness_exit_block_is_removed(wiki: nx.MultiDiGraph):
  liveness.AnnotateLiveness(wiki, "b1")
  assert "__liveness_starting_point__" not in wiki


def test_AnnotateLiveness_wiki_b1(wiki: nx.MultiDiGraph):
  """Test liveness annotations from block b1."""
  live_variable_count, data_flow_steps = liveness.AnnotateLiveness(wiki, "b1")
  assert live_variable_count == 3
  assert data_flow_steps == 5

  # Features:
  assert wiki.nodes["b1"]["x"] == [-1, 1]
  assert wiki.nodes["b2"]["x"] == [-1, 0]
  assert wiki.nodes["b3a"]["x"] == [-1, 0]
  assert wiki.nodes["b3b"]["x"] == [-1, 0]

  assert wiki.nodes["a"]["x"] == [-1, 0]
  assert wiki.nodes["b"]["x"] == [-1, 0]
  assert wiki.nodes["c"]["x"] == [-1, 0]
  assert wiki.nodes["d"]["x"] == [-1, 0]

  # Labels:
  assert not wiki.nodes["b1"]["y"]
  assert not wiki.nodes["b2"]["y"]
  assert not wiki.nodes["b3a"]["y"]
  assert not wiki.nodes["b3b"]["y"]

  assert wiki.nodes["a"]["y"]
  assert wiki.nodes["b"]["y"]
  assert not wiki.nodes["c"]["y"]
  assert wiki.nodes["d"]["y"]


def test_AnnotateLiveness_wiki_b2(wiki: nx.MultiDiGraph):
  """Test liveness annotations from block b2."""
  live_variable_count, data_flow_steps = liveness.AnnotateLiveness(wiki, "b2")
  assert live_variable_count == 2
  assert data_flow_steps == 5

  # Features:
  assert wiki.nodes["b1"]["x"] == [-1, 0]
  assert wiki.nodes["b2"]["x"] == [-1, 1]
  assert wiki.nodes["b3a"]["x"] == [-1, 0]
  assert wiki.nodes["b3b"]["x"] == [-1, 0]

  assert wiki.nodes["a"]["x"] == [-1, 0]
  assert wiki.nodes["b"]["x"] == [-1, 0]
  assert wiki.nodes["c"]["x"] == [-1, 0]
  assert wiki.nodes["d"]["x"] == [-1, 0]

  # Labels:
  assert not wiki.nodes["b1"]["y"]
  assert not wiki.nodes["b2"]["y"]
  assert not wiki.nodes["b3a"]["y"]
  assert not wiki.nodes["b3b"]["y"]

  assert not wiki.nodes["a"]["y"]
  assert wiki.nodes["b"]["y"]
  assert not wiki.nodes["c"]["y"]
  assert wiki.nodes["d"]["y"]


def test_AnnotateLiveness_wiki_b3a(wiki: nx.MultiDiGraph):
  """Test liveness annotations from block b3a."""
  live_variable_count, data_flow_steps = liveness.AnnotateLiveness(wiki, "b3a")
  assert live_variable_count == 3
  assert data_flow_steps == 5

  # Features:
  assert wiki.nodes["b1"]["x"] == [-1, 0]
  assert wiki.nodes["b2"]["x"] == [-1, 0]
  assert wiki.nodes["b3a"]["x"] == [-1, 1]
  assert wiki.nodes["b3b"]["x"] == [-1, 0]

  assert wiki.nodes["a"]["x"] == [-1, 0]
  assert wiki.nodes["b"]["x"] == [-1, 0]
  assert wiki.nodes["c"]["x"] == [-1, 0]
  assert wiki.nodes["d"]["x"] == [-1, 0]

  # Labels:
  assert not wiki.nodes["b1"]["y"]
  assert not wiki.nodes["b2"]["y"]
  assert not wiki.nodes["b3a"]["y"]
  assert not wiki.nodes["b3b"]["y"]

  assert not wiki.nodes["a"]["y"]
  assert wiki.nodes["b"]["y"]
  assert wiki.nodes["c"]["y"]
  assert wiki.nodes["d"]["y"]


def test_AnnotateLiveness_wiki_b3b(wiki: nx.MultiDiGraph):
  """Test liveness annotations from block b3b."""
  live_variable_count, data_flow_steps = liveness.AnnotateLiveness(wiki, "b3b")
  assert live_variable_count == 0
  assert data_flow_steps == 5

  # Features:
  assert wiki.nodes["b1"]["x"] == [-1, 0]
  assert wiki.nodes["b2"]["x"] == [-1, 0]
  assert wiki.nodes["b3a"]["x"] == [-1, 0]
  assert wiki.nodes["b3b"]["x"] == [-1, 1]

  assert wiki.nodes["a"]["x"] == [-1, 0]
  assert wiki.nodes["b"]["x"] == [-1, 0]
  assert wiki.nodes["c"]["x"] == [-1, 0]
  assert wiki.nodes["d"]["x"] == [-1, 0]

  # Labels:
  assert not wiki.nodes["b1"]["y"]
  assert not wiki.nodes["b2"]["y"]
  assert not wiki.nodes["b3a"]["y"]
  assert not wiki.nodes["b3b"]["y"]

  assert not wiki.nodes["a"]["y"]
  assert not wiki.nodes["b"]["y"]
  assert not wiki.nodes["c"]["y"]
  assert not wiki.nodes["d"]["y"]


@test.Fixture(scope="function")
def graph() -> nx.MultiDiGraph:
  g = nx.MultiDiGraph()
  g.add_node("A", type="statement", x=-1)
  g.add_node("B", type="statement", x=-1)
  g.add_node("C", type="statement", x=-1)
  g.add_node("D", type="statement", x=-1)

  g.add_node("%1", type="identifier", x=-1)
  g.add_node("%2", type="identifier", x=-1)

  g.add_edge("A", "B", flow="control")
  g.add_edge("A", "C", flow="control")
  g.add_edge("B", "D", flow="control")
  g.add_edge("C", "D", flow="control")

  g.add_edge("A", "%1", flow="data")
  g.add_edge("%1", "B", flow="data")
  g.add_edge("B", "%2", flow="data")
  g.add_edge("%2", "D", flow="data")

  return g


def test_AnnotateDominatorTree_graph_A(graph: nx.MultiDiGraph):
  live_variable_count, data_flow_steps = liveness.AnnotateLiveness(graph, "A")
  assert live_variable_count == 2
  assert data_flow_steps == 4

  # Features:
  assert graph.nodes["A"]["x"] == [-1, 1]
  assert graph.nodes["B"]["x"] == [-1, 0]
  assert graph.nodes["C"]["x"] == [-1, 0]
  assert graph.nodes["D"]["x"] == [-1, 0]

  assert graph.nodes["%1"]["x"] == [-1, 0]
  assert graph.nodes["%2"]["x"] == [-1, 0]

  # Labels:
  assert not graph.nodes["A"]["y"]
  assert not graph.nodes["B"]["y"]
  assert not graph.nodes["C"]["y"]
  assert not graph.nodes["D"]["y"]

  assert graph.nodes["%1"]["y"]
  assert graph.nodes["%2"]["y"]


def test_AnnotateDominatorTree_graph_B(graph: nx.MultiDiGraph):
  live_variable_count, data_flow_steps = liveness.AnnotateLiveness(graph, "B")
  assert live_variable_count == 1
  assert data_flow_steps == 4

  # Features:
  assert graph.nodes["A"]["x"] == [-1, 0]
  assert graph.nodes["B"]["x"] == [-1, 1]
  assert graph.nodes["C"]["x"] == [-1, 0]
  assert graph.nodes["D"]["x"] == [-1, 0]

  assert graph.nodes["%1"]["x"] == [-1, 0]
  assert graph.nodes["%2"]["x"] == [-1, 0]

  # Labels:
  assert not graph.nodes["A"]["y"]
  assert not graph.nodes["B"]["y"]
  assert not graph.nodes["C"]["y"]
  assert not graph.nodes["D"]["y"]

  assert not graph.nodes["%1"]["y"]
  assert graph.nodes["%2"]["y"]


def test_AnnotateDominatorTree_graph_C(graph: nx.MultiDiGraph):
  live_variable_count, data_flow_steps = liveness.AnnotateLiveness(graph, "C")
  assert live_variable_count == 1
  assert data_flow_steps == 4

  # Features:
  assert graph.nodes["A"]["x"] == [-1, 0]
  assert graph.nodes["B"]["x"] == [-1, 0]
  assert graph.nodes["C"]["x"] == [-1, 1]
  assert graph.nodes["D"]["x"] == [-1, 0]

  assert graph.nodes["%1"]["x"] == [-1, 0]
  assert graph.nodes["%2"]["x"] == [-1, 0]

  # Labels:
  assert not graph.nodes["A"]["y"]
  assert not graph.nodes["B"]["y"]
  assert not graph.nodes["C"]["y"]
  assert not graph.nodes["D"]["y"]

  assert not graph.nodes["%1"]["y"]
  assert graph.nodes["%2"]["y"]


def test_AnnotateDominatorTree_graph_D(graph: nx.MultiDiGraph):
  live_variable_count, data_flow_steps = liveness.AnnotateLiveness(graph, "D")
  assert live_variable_count == 0
  assert data_flow_steps == 4

  # Features:
  assert graph.nodes["A"]["x"] == [-1, 0]
  assert graph.nodes["B"]["x"] == [-1, 0]
  assert graph.nodes["C"]["x"] == [-1, 0]
  assert graph.nodes["D"]["x"] == [-1, 1]

  assert graph.nodes["%1"]["x"] == [-1, 0]
  assert graph.nodes["%2"]["x"] == [-1, 0]

  # Labels:
  assert not graph.nodes["A"]["y"]
  assert not graph.nodes["B"]["y"]
  assert not graph.nodes["C"]["y"]
  assert not graph.nodes["D"]["y"]

  assert not graph.nodes["%1"]["y"]
  assert not graph.nodes["%2"]["y"]


@test.Fixture(scope="function")
def while_loop() -> nx.MultiDiGraph:
  """Test fixture which returns a simple "while loop" graph."""
  #          (%1)
  #            |
  #            V
  #    +------[A]<--------+
  #    |       |          |
  #    |       V          |
  #    |      [Ba]----+   |
  #    |       |      |   |
  #    |       |      V   |
  #    |       |     (%3) |
  #    |       V      |   |
  #    |    +-[Bb]<---+   |
  #    |    |  |          |
  #    |    |  +----------+
  #    |  (%2)
  #    V    |
  #   [C]<--+
  g = nx.MultiDiGraph()
  g.add_node("A", type="statement", x=-1)  # Loop header
  g.add_node("Ba", type="statement", x=-1)  # Loop body
  g.add_node("Bb", type="statement", x=-1)  # Loop body
  g.add_node("C", type="statement", x=-1)  # Loop exit

  # Control flow:
  g.add_edge("A", "Ba", flow="control")
  g.add_edge("Ba", "Bb", flow="control")
  g.add_edge("Bb", "A", flow="control")
  g.add_edge("A", "C", flow="control")

  # Data flow:
  g.add_node("%1", type="identifier", x=-1)  # Loop induction variable
  g.add_node("%2", type="identifier", x=-1)  # Computed result
  g.add_node("%3", type="identifier", x=-1)  # Intermediate value

  g.add_edge("%1", "A", flow="data")
  g.add_edge("Ba", "%3", flow="data")
  g.add_edge("%3", "Bb", flow="data")
  g.add_edge("Bb", "%2", flow="data")
  g.add_edge("%2", "C", flow="data")

  return g


def test_AnnotateDominatorTree_while_loop_A(while_loop: nx.MultiDiGraph):
  live_variable_count, data_flow_steps = liveness.AnnotateLiveness(
    while_loop, "A"
  )
  assert live_variable_count == 2
  assert data_flow_steps == 5

  # Features:
  assert while_loop.nodes["A"]["x"] == [-1, 1]
  assert while_loop.nodes["Ba"]["x"] == [-1, 0]
  assert while_loop.nodes["Bb"]["x"] == [-1, 0]
  assert while_loop.nodes["C"]["x"] == [-1, 0]

  assert while_loop.nodes["%1"]["x"] == [-1, 0]
  assert while_loop.nodes["%2"]["x"] == [-1, 0]
  assert while_loop.nodes["%3"]["x"] == [-1, 0]

  # Labels:
  assert not while_loop.nodes["A"]["y"]
  assert not while_loop.nodes["Ba"]["y"]
  assert not while_loop.nodes["Bb"]["y"]
  assert not while_loop.nodes["C"]["y"]

  assert while_loop.nodes["%1"]["y"]  # Loop induction variable
  assert while_loop.nodes["%2"]["y"]  # Computed result
  assert not while_loop.nodes["%3"]["y"]  # Intermediate value


def test_AnnotateDominatorTree_while_loop_Ba(while_loop: nx.MultiDiGraph):
  live_variable_count, data_flow_steps = liveness.AnnotateLiveness(
    while_loop, "Ba"
  )
  assert live_variable_count == 2
  assert data_flow_steps == 5

  # Features:
  assert while_loop.nodes["A"]["x"] == [-1, 0]
  assert while_loop.nodes["Ba"]["x"] == [-1, 1]
  assert while_loop.nodes["Bb"]["x"] == [-1, 0]
  assert while_loop.nodes["C"]["x"] == [-1, 0]

  assert while_loop.nodes["%1"]["x"] == [-1, 0]
  assert while_loop.nodes["%2"]["x"] == [-1, 0]
  assert while_loop.nodes["%3"]["x"] == [-1, 0]

  # Labels:
  assert not while_loop.nodes["A"]["y"]
  assert not while_loop.nodes["Ba"]["y"]
  assert not while_loop.nodes["Bb"]["y"]
  assert not while_loop.nodes["C"]["y"]

  assert while_loop.nodes["%1"]["y"]  # Loop induction variable
  assert not while_loop.nodes["%2"]["y"]  # Computed result
  assert while_loop.nodes["%3"]["y"]  # Intermediate value


def test_AnnotateDominatorTree_while_loop_Bb(while_loop: nx.MultiDiGraph):
  live_variable_count, data_flow_steps = liveness.AnnotateLiveness(
    while_loop, "Bb"
  )
  assert live_variable_count == 2
  assert data_flow_steps == 5

  # Features:
  assert while_loop.nodes["A"]["x"] == [-1, 0]
  assert while_loop.nodes["Ba"]["x"] == [-1, 0]
  assert while_loop.nodes["Bb"]["x"] == [-1, 1]
  assert while_loop.nodes["C"]["x"] == [-1, 0]

  assert while_loop.nodes["%1"]["x"] == [-1, 0]
  assert while_loop.nodes["%2"]["x"] == [-1, 0]
  assert while_loop.nodes["%3"]["x"] == [-1, 0]

  # Labels:
  assert not while_loop.nodes["A"]["y"]
  assert not while_loop.nodes["Ba"]["y"]
  assert not while_loop.nodes["Bb"]["y"]
  assert not while_loop.nodes["C"]["y"]

  assert while_loop.nodes["%1"]["y"]  # Loop induction variable
  assert while_loop.nodes["%2"]["y"]  # Computed result
  assert not while_loop.nodes["%3"]["y"]  # Intermediate value


def test_AnnotateDominatorTree_while_loop_C(while_loop: nx.MultiDiGraph):
  live_variable_count, data_flow_steps = liveness.AnnotateLiveness(
    while_loop, "C"
  )
  assert live_variable_count == 0
  assert data_flow_steps == 5

  # Features:
  assert while_loop.nodes["A"]["x"] == [-1, 0]
  assert while_loop.nodes["Ba"]["x"] == [-1, 0]
  assert while_loop.nodes["Bb"]["x"] == [-1, 0]
  assert while_loop.nodes["C"]["x"] == [-1, 1]

  assert while_loop.nodes["%1"]["x"] == [-1, 0]
  assert while_loop.nodes["%2"]["x"] == [-1, 0]
  assert while_loop.nodes["%3"]["x"] == [-1, 0]

  # Labels:
  assert not while_loop.nodes["A"]["y"]
  assert not while_loop.nodes["Ba"]["y"]
  assert not while_loop.nodes["Bb"]["y"]
  assert not while_loop.nodes["C"]["y"]

  assert not while_loop.nodes["%1"]["y"]  # Loop induction variable
  assert not while_loop.nodes["%2"]["y"]  # Computed result
  assert not while_loop.nodes["%3"]["y"]  # Intermediate value


if __name__ == "__main__":
  test.Main()
