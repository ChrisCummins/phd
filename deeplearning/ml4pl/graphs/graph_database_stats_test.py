"""Unit tests for //deeplearning/ml4pl/ggnn:graph_database."""
import pathlib
import pickle

import numpy as np

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs import graph_database_stats as stats
from deeplearning.ml4pl.graphs.labelled import graph_tuple
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def db(tempdir: pathlib.Path) -> graph_database.Database:
  """Fixture that returns an sqlite database."""
  yield graph_database.Database(f"sqlite:///{tempdir}/db")


@test.Fixture(scope="function")
def db_512(db: graph_database.Database) -> graph_database.Database:
  """Fixture which returns a database with 512 graphs, indexed by node_count."""

  def _MakeGraphMeta(i):
    return graph_database.GraphMeta(
      group="train",
      bytecode_id=1,
      source_name="foo",
      relpath="bar",
      language="c",
      node_count=i,
      edge_count=2,
      graph_labels_dimensionality=1,
      loop_connectedness=0,
      edge_position_max=0,
      undirected_diameter=0,
      graph=graph_database.Graph(
        pickled_data=pickle.dumps(
          graph_tuple.GraphTuple(
            adjacency_lists="unused",
            edge_positions="unused",
            incoming_edge_counts="unused",
            node_x_indices="unused",
            graph_y=np.array([1, 2, 3], dtype=np.float32),
          )
        )
      ),
    )

  with db.Session(commit=True) as s:
    s.add_all([_MakeGraphMeta(i) for i in range(512)])

  return db


def test_GraphDatabaseStats_graph_count(db_512: graph_database.Database):
  """Test that the expected number of graphs are returned"""
  s = stats.GraphDatabaseStats(db_512)
  # Note that the first two rows (with zero and one node respectively) are
  # ignored by the database reader.
  assert s.graph_count == 510


def test_GraphDatabaseStats_repr(db_512: graph_database.Database):
  """Test the string representation of the stats object"""
  s = stats.GraphDatabaseStats(db_512)
  assert str(s) == (
    "Graphs database: 510 instances, 1 edge type, "
    "(8568x200, 2x2) float64 node embeddings, "
    "1-d graph labels, max 511 nodes, max 2 edges, "
    "0 max edge positions"
  )


def test_GraphTupleDatabaseStats_repr(db_512: graph_database.Database):
  """Test the string representation of the stats object"""
  s = stats.GraphTupleDatabaseStats(db_512)
  assert str(s) == (
    "Graphs database: 510 instances, 1 edge type, "
    "(8568x200, 2x2) float64 node embeddings, 1-d float32 graph labels, "
    "max 511 nodes, max 2 edges, 0 max edge positions"
  )


def test_GraphDatabaseStats_groups(db_512: graph_database.Database):
  """Test that a list of distinct group names is returned."""
  s = stats.GraphTupleDatabaseStats(db_512)
  assert s.groups == ["train"]


if __name__ == "__main__":
  test.Main()
