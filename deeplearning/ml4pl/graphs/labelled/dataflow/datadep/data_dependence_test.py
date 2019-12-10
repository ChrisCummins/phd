"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow/datadep:data_dependence."""
import networkx as nx

from deeplearning.ml4pl.graphs.labelled.dataflow.datadep import data_dependence
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


def test_AnnotateDominatorTree():
  g = nx.MultiDiGraph()
  g.add_node("A", type="statement", x=-1)
  g.add_node("B", type="statement", x=-1)
  g.add_node("C", type="statement", x=-1)
  g.add_node("D", type="statement", x=-1)
  g.add_node("E", type="statement", x=-1)

  g.add_edge("A", "B", flow="control")
  g.add_edge("A", "C", flow="control")
  g.add_edge("B", "D", flow="control")
  g.add_edge("C", "D", flow="control")
  g.add_edge("A", "E", flow="control")

  g.add_edge("A", "B", flow="data")
  g.add_edge("A", "C", flow="data")
  g.add_edge("C", "D", flow="data")

  dependent_node_count, max_steps = data_dependence.AnnotateDataDependencies(
    g, "D"
  )
  assert dependent_node_count == 3
  assert max_steps == 3

  # Features
  assert g.nodes["A"]["x"] == [-1, 0]
  assert g.nodes["B"]["x"] == [-1, 0]
  assert g.nodes["C"]["x"] == [-1, 0]
  assert g.nodes["D"]["x"] == [-1, 1]
  assert g.nodes["E"]["x"] == [-1, 0]

  # Labels
  assert g.nodes["A"]["y"]
  assert not g.nodes["B"]["y"]
  assert g.nodes["C"]["y"]
  assert g.nodes["D"]["y"]
  assert not g.nodes["E"]["y"]


if __name__ == "__main__":
  test.Main()
