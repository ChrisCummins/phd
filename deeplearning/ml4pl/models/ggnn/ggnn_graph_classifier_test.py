"""Unit tests for //deeplearning/ml4pl/models/ggnn:ggnn_graph_classifier."""
import pathlib
import pickle
import pytest

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.models.ggnn import ggnn_node_classifier
from labm8 import app
from labm8 import test


FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def db(tempdir: pathlib.Path) -> graph_database.Database:
  """A test fixture that yields an empty database."""
  return graph_database.Database(f'sqlite:///{tempdir}/db')


def test_Train(db: graph_database.Database):
  FLAGS.num_epochs = 2

  with db.Session(commit=True) as session:
    session.add_all([
      graph_database.GraphMeta(
          group="train",
          bytecode_id=0,
          source_name="source",
          relpath="relpath",
          language="c",
          node_count=3,
          edge_count=2,
          edge_type_count=3,
          edge_features_dimensionality=1,
          graphs_labels_dimensionality=1,
          graph=graph_database.Graph(
              data=pickle.dumps({
                'adjacency_lists': [[(0, 1)], [(1, 2)], []],
                'edge_x': [[1], [2]],
                'graph_y': [1],
              }))),
      graph_database.GraphMeta(
          group="train",
          bytecode_id=0,
          source_name="source",
          relpath="relpath",
          language="c",
          node_count=3,
          edge_count=2,
          edge_type_count=3,
          edge_features_dimensionality=1,
          graphs_labels_dimensionality=1,
          graph=graph_database.Graph(
              data=pickle.dumps({
                'adjacency_lists': [[(0, 1)], [(1, 2)], []],
                'edge_x': [[1], [2]],
                'graph_y': [1],
              }))),
      graph_database.GraphMeta(
          group="train",
          bytecode_id=0,
          source_name="source",
          relpath="relpath",
          language="c",
          node_count=3,
          edge_count=2,
          edge_type_count=3,
          edge_features_dimensionality=1,
          graphs_labels_dimensionality=1,
          graph=graph_database.Graph(
              data=pickle.dumps({
                'adjacency_lists': [[(0, 1)], [(1, 2)], []],
                'edge_x': [[1], [2]],
                'graph_y': [1],
              }))),
    ])

  model = ggnn_node_classifier.GgnnNodeClassifierModel(db_3)
  model.Train()


if __name__ == '__main__':
  test.Main()
