"""Unit tests for //deeplearning/ml4pl/models/ggnn:ggnn_node_classifier."""
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
  """A test fixture that yields a database with 3 rows."""
  return graph_database.Database(f'sqlite:///{tempdir}/db')


def test_Train_2d_node_features_2d_node_labels(db: graph_database.Database):
  """Test training a model with a graph dataset with 2D node features and
  labels and a single edge type.

  This is similar to the type of dataset that would be used for learning control
  flow reachability.
  """
  FLAGS.num_epochs = 2

  with db.Session(commit=True) as session:
    session.add_all([
      graph_database.GraphMeta(
          group="train",
          bytecode_id=0,
          source_name="source",
          relpath="relpath",
          language="c",
          edge_type_count=1,
          node_count=3,
          edge_count=2,
          node_features_dimensionality=2,
          node_labels_dimensionality=2,
          data_flow_max_steps_required=2,
          graph=graph_database.Graph(
              data=pickle.dumps({
                'adjacency_lists': [[(0, 1), (1, 2)]],
                'node_x': [
                  (1, 0),
                  (0, 1),
                  (0, 0),
                ],
                'node_y': [
                  (1, 0),
                  (0, 1),
                  (0, 0),
                ],
              }))),
      graph_database.GraphMeta(
          group="val",
          bytecode_id=0,
          source_name="source",
          relpath="relpath",
          language="c",
          edge_type_count=1,
          node_count=3,
          edge_count=2,
          node_features_dimensionality=2,
          node_labels_dimensionality=2,
          data_flow_max_steps_required=2,
          graph=graph_database.Graph(
              data=pickle.dumps({
                'adjacency_lists': [[(0, 1), (1, 2)]],
                'node_x': [
                  (1, 0),
                  (0, 1),
                  (0, 0),
                ],
                'node_y': [
                  (1, 0),
                  (0, 1),
                  (0, 0),
                ],
              }))),
      graph_database.GraphMeta(
          group="test",
          bytecode_id=0,
          source_name="source",
          relpath="relpath",
          language="c",
          edge_type_count=1,
          node_count=3,
          edge_count=2,
          node_features_dimensionality=2,
          node_labels_dimensionality=2,
          data_flow_max_steps_required=2,
          graph=graph_database.Graph(
              data=pickle.dumps({
                'adjacency_lists': [[(0, 1), (1, 2)]],
                'node_x': [
                  (1, 0),
                  (0, 1),
                  (0, 0),
                ],
                'node_y': [
                  (1, 0),
                  (0, 1),
                  (0, 0),
                ],
              }))),
    ])

  model = ggnn_node_classifier.GgnnNodeClassifierModel(db)
  model.Train()


if __name__ == '__main__':
  test.Main()
