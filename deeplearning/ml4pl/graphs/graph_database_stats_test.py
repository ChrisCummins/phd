"""Unit tests for //deeplearning/ml4pl/ggnn:graph_database."""
import pathlib
import pickle

import numpy as np
import pytest
from labm8 import app
from labm8 import test

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs import graph_database_stats as stats
from deeplearning.ml4pl.graphs.labelled.graph_tuple import graph_tuple

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def db(tempdir: pathlib.Path) -> graph_database.Database:
  """Fixture that returns an sqlite database."""
  yield graph_database.Database(f'sqlite:///{tempdir}/db')


@pytest.fixture(scope='function')
def db_512(db: graph_database.Database) -> graph_database.Database:
  """Fixture which returns a database with 512 graphs, indexed by node_count."""

  def _MakeGraphMeta(i):
    return graph_database.GraphMeta(
        group="train",
        bytecode_id=1,
        source_name='foo',
        relpath='bar',
        language='c',
        node_count=i,
        edge_count=2,
        graph_labels_dimensionality=1,
        loop_connectedness=0,
        edge_position_max=0,
        undirected_diameter=0,
        graph=graph_database.Graph(pickled_data=pickle.dumps(
            graph_tuple.GraphTuple(adjacency_lists='unused',
                                   edge_positions='unused',
                                   incoming_edge_counts='unused',
                                   node_x_indices='unused',
                                   graph_y=np.array([1, 2, 3],
                                                    dtype=np.float32)))))

  with db.Session(commit=True) as s:
    s.add_all([_MakeGraphMeta(i) for i in range(512)])

  return db


def test_GraphDatabaseStats_graph_count(db_512: graph_database.Database):
  """Test that the expected number of graphs are returned"""
  s = stats.GraphDatabaseStats(db_512)
  assert s.graph_count == 512


def test_GraphDatabaseStats_repr(db_512: graph_database.Database):
  """Test the string representation of the stats object"""
  s = stats.GraphDatabaseStats(db_512)
  assert str(s) == ("Graphs database: 512 instances, 1 edge type, "
                    "8568x200 float64 node embeddings, 1-d graph labels, "
                    "max 511 nodes, max 2 edges")


def test_GraphTupleDatabaseStats_repr(db_512: graph_database.Database):
  """Test the string representation of the stats object"""
  s = stats.GraphTupleDatabaseStats(db_512)
  assert str(s) == (
      "Graphs database: 512 instances, 1 edge type, "
      "8568x200 float64 node embeddings, 1-d float32 graph labels, "
      "max 511 nodes, max 2 edges")


if __name__ == '__main__':
  test.Main()
