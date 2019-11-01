"""Unit tests for //deeplearning/ml4pl/models/ggnn:ggnn_base."""
import pathlib
import pickle

import numpy as np
import pytest
import tensorflow as tf
from labm8 import app
from labm8 import test

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled.graph_tuple import \
  graph_tuple as graph_tuples
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.ggnn import ggnn_base

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def graph_db(tempdir: pathlib.Path) -> graph_database.Database:
  db = graph_database.Database(f'sqlite:///{tempdir}/graphs.db')
  with db.Session(commit=True) as s:
    s.add(
        graph_database.GraphMeta(
            group="train",
            bytecode_id=0,
            source_name="source",
            relpath="relpath",
            language="c",
            node_count=3,
            edge_count=2,
            edge_type_count=3,
            graph_labels_dimensionality=1,
            edge_position_max=0,
            loop_connectedness=0,
            undirected_diameter=1,
            graph=graph_database.Graph(pickled_data=pickle.dumps(
                graph_tuples.GraphTuple(
                    adjacency_lists=None,
                    edge_positions=None,
                    incoming_edge_counts=None,
                    node_x_indices=None,
                    graph_y=np.array(np.array([1], dtype=np.float32)),
                )))))
  return db


@pytest.fixture(scope='function')
def log_db(tempdir: pathlib.Path) -> log_database.Database:
  return log_database.Database(f'sqlite:///{tempdir}/logs.db')


class MockModel(ggnn_base.GgnnBaseModel):
  """A mock GGNN model."""

  def MakeLossAndAccuracyAndPredictionOps(self):
    self.placeholders["X"] = tf.compat.v1.placeholder("float")
    self.placeholders["Y"] = tf.compat.v1.placeholder("float")
    W = tf.Variable(np.random.randn(), name="weight")
    b = tf.Variable(np.random.randn(), name="bias")
    predictions = tf.add(tf.multiply(self.placeholders["X"], W), b)
    loss = tf.reduce_sum(tf.pow(predictions - self.placeholders["Y"], 2))
    accuracy = tf.Variable(np.random.randn())
    accuracies = tf.Variable(np.random.rand())
    return loss, accuracy, accuracies, predictions

  def MakeMinibatchIterator(self, group):
    del epoch_type
    for _ in range(3):
      log = log_database.BatchLog(graph_count=10)
      yield log, {
          self.placeholders["X"]: 5,
          self.placeholders["Y"]: 10,
      }


def test_SaveModel(tempdir: pathlib.Path, tempdir2: pathlib.Path,
                   graph_db: graph_database.Database,
                   log_db: log_database.Database):
  """Test saving a model to file."""
  FLAGS.working_dir = tempdir2

  model = MockModel(graph_db, log_db)
  model.InitializeModel()
  model.global_training_step = 10
  model.SaveModel(tempdir / 'foo.pickle')
  assert (tempdir / 'foo.pickle').is_file()

  with open(tempdir / 'foo.pickle', 'rb') as f:
    saved_model = pickle.load(f)

  assert 'model_flags' in saved_model
  assert 'model_data' in saved_model
  assert saved_model['global_training_step'] == 10


def test_LoadModel(tempdir: pathlib.Path, tempdir2: pathlib.Path,
                   graph_db: graph_database.Database,
                   log_db: log_database.Database):
  """Test loading a model from file."""
  FLAGS.working_dir = tempdir2

  model = MockModel(graph_db, log_db)
  model.InitializeModel()
  model.epoch_num = 2
  model.global_training_step = 10
  model.SaveModel(tempdir / 'foo.pickle')

  model.epoch_num = 0
  model.global_training_step = 0
  model.LoadModel(tempdir / 'foo.pickle')
  assert model.epoch_num == 2
  assert model.global_training_step == 10


def test_LoadModel_unknown_saved_model_flag(
    tempdir: pathlib.Path, tempdir2: pathlib.Path,
    graph_db: graph_database.Database, log_db: log_database.Database):
  """Test that error is raised if saved model contains unknown flag."""
  FLAGS.working_dir = tempdir2
  model = MockModel(graph_db, log_db)
  model.InitializeModel()
  model.SaveModel(tempdir / 'foo.pickle')

  with open(tempdir / 'foo.pickle', 'rb') as f:
    saved_model = pickle.load(f)

  saved_model['model_flags']['a new flag'] = 10

  with open(tempdir / 'foo.pickle', 'wb') as f:
    pickle.dump(saved_model, f)

  with pytest.raises(EnvironmentError) as e_ctx:
    model.LoadModel(tempdir / 'foo.pickle')

  assert 'a new flag' in str(e_ctx.value)


def test_Train(tempdir2: pathlib.Path, graph_db: graph_database.Database,
               log_db: log_database.Database):
  """Test that training terminates and bumps the epoch number."""
  FLAGS.working_dir = tempdir2
  FLAGS.num_epochs = 1

  model = MockModel(graph_db, log_db)
  model.InitializeModel()
  model.Train()
  assert model.best_epoch_num == 1


if __name__ == '__main__':
  test.Main()
