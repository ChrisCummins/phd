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
def db_3(tempdir: pathlib.Path) -> graph_database.Database:
  """A test fixture that yields a database with 3 rows.

  One GraphMeta and Graph each for train/val/test.
  """
  db = graph_database.Database(f'sqlite:///{tempdir}/db')
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
            node_features_dimensionality=5,
            node_labels_dimensionality=3,
            data_flow_max_steps_required=2,
            graph=graph_database.Graph(
                data=pickle.dumps({
                    'adjacency_list': [(0, 1), (1, 2)],
                    'node_features': [
                        (1, 0, 0, 0, 0),
                        (0, 1, 0, 0, 0),
                        (0, 0, 1, 0, 0),
                    ],
                    'targets': [
                        (1, 0, 0),
                        (0, 1, 0),
                        (0, 0, 1),
                    ],
                }))),
        graph_database.GraphMeta(
            group="val",
            bytecode_id=0,
            source_name="source",
            relpath="relpath",
            language="c",
            node_count=3,
            edge_count=2,
            node_features_dimensionality=5,
            node_labels_dimensionality=3,
            data_flow_max_steps_required=2,
            graph=graph_database.Graph(
                data=pickle.dumps({
                    'adjacency_list': [(0, 1), (1, 2)],
                    'node_features': [
                        (1, 0, 0, 0, 0),
                        (0, 1, 0, 0, 0),
                        (0, 0, 1, 0, 0),
                    ],
                    'targets': [
                        (1, 0, 0),
                        (0, 1, 0),
                        (0, 0, 1),
                    ],
                }))),
        graph_database.GraphMeta(
            group="test",
            bytecode_id=0,
            source_name="source",
            relpath="relpath",
            language="c",
            node_count=3,
            edge_count=2,
            node_features_dimensionality=5,
            node_labels_dimensionality=3,
            data_flow_max_steps_required=2,
            graph=graph_database.Graph(
                data=pickle.dumps({
                    'adjacency_list': [(0, 1), (1, 2)],
                    'node_features': [
                        (1, 0, 0, 0, 0),
                        (0, 1, 0, 0, 0),
                        (0, 0, 1, 0, 0),
                    ],
                    'targets': [
                        (1, 0, 0),
                        (0, 1, 0),
                        (0, 0, 1),
                    ],
                }))),
    ])
  return db


def test_Train(db_3: graph_database.Database):
  FLAGS.num_epochs = 2
  model = ggnn_node_classifier.GgnnNodeClassifierModel(db_3)
  model.Train()


if __name__ == '__main__':
  test.Main()
