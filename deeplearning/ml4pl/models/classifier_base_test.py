"""Unit tests for //deeplearning/ml4pl/models:classifier_base."""
import pathlib
import pickle
import typing

import networkx as nx
import numpy as np
import pytest
from labm8 import app
from labm8 import test

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.unlabelled.cdfg import random_cdfg_generator
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def graph_db(tempdir: pathlib.Path) -> graph_database.Database:
  """Fixture which returns a graph database containing 50 random graphs."""
  db = graph_database.Database(f'sqlite:///{tempdir}/graphs.db')
  graphs = (list(_MakeNRandomGraphs(30, 'train')) +
            list(_MakeNRandomGraphs(10, 'val')) +
            list(_MakeNRandomGraphs(10, 'test')))
  random_cdfg_generator.AddRandomAnnotations(graphs,
                                             graph_y_choices=[
                                                 np.array([1, 0],
                                                          dtype=np.int32),
                                                 np.array([0, 1],
                                                          dtype=np.int32)
                                             ])
  with db.Session(commit=True) as s:
    s.add_all([graph_database.GraphMeta.CreateFromNetworkX(g) for g in graphs])
  return db


@pytest.fixture(scope='function')
def log_db(tempdir: pathlib.Path) -> log_database.Database:
  return log_database.Database(f'sqlite:///{tempdir}/logs.db')


class MockModel(classifier_base.ClassifierBase):
  """A mock model."""

  def __init__(self, *args, **kwargs):
    super(MockModel, self).__init__(*args, **kwargs)
    self.mock_data = 1

  def ModelDataToSave(self):
    """Prepare data to save."""
    return {"mock_data": self.mock_data}

  def LoadModelData(self, data_to_load):
    """Reset model state from loaded data."""
    self.mock_data = data_to_load["mock_data"]

  def MakeMinibatchIterator(
      self, epoch_type: str, group: str
  ) -> typing.Iterable[typing.Tuple[log_database.BatchLogMeta, typing.Any]]:
    """Generate mini-batches of fake data."""
    for i in range(10):
      log = log_database.BatchLogMeta(
          group=group,
          type=epoch_type,
          node_count=10,
          graph_count=10,  # fake the number of graphs as this is checked.
          batch_log=log_database.BatchLog(),
      )
      log.graph_indices = []
      yield log, i

  def RunMinibatch(self, log: log_database.BatchLogMeta,
                   i: int) -> classifier_base.ClassifierBase.MinibatchResults:
    """Fake mini-batch 'run'."""
    log.loss = 0
    return classifier_base.ClassifierBase.MinibatchResults(
        y_true_1hot=np.array([np.array([1], dtype=np.int32)]),
        y_pred_1hot=np.array([np.array([0], dtype=np.int32)]),
    )


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
  model.mock_data = 100
  model.SaveModel(tempdir / 'foo.pickle')

  model.epoch_num = 0
  model.mock_data = 0
  model.global_training_step = 0
  model.LoadModel(tempdir / 'foo.pickle')
  assert model.epoch_num == 2
  assert model.global_training_step == 10
  assert model.mock_data == 100


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

  model = MockModel(graph_db, log_db)
  model.InitializeModel()
  model.Train(num_epochs=1)
  assert model.best_epoch_num == 1


def _MakeNRandomGraphs(n: int, group: str) -> typing.Iterable[nx.MultiDiGraph]:
  """Private helper to generate random graphs of the given group."""
  for i in range(n):
    g = random_cdfg_generator.FastCreateRandom()
    g.bytecode_id = 0
    g.relpath = str(i)
    g.language = 'c'
    g.group = group
    g.source_name = 'rand'
    yield g


if __name__ == '__main__':
  test.Main()
