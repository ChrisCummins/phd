"""Unit tests for //deeplearning/ml4pl/models:classifier_base."""
import pathlib
import pickle
import typing

import networkx as nx
import numpy as np
import pytest
import sqlalchemy as sql

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.unlabelled.cdfg import random_cdfg_generator
from deeplearning.ml4pl.models import classifier_base
from deeplearning.ml4pl.models import log_database
from labm8.py import app
from labm8.py import test

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
      self, epoch_type: str, groups: typing.List[str], print_context=None
  ) -> typing.Iterable[typing.Tuple[log_database.BatchLogMeta, typing.Any]]:
    """Generate mini-batches of fake data."""
    for group in groups:
      for i in range(10):
        log = log_database.BatchLogMeta(
            group=group,
            type=epoch_type,
            node_count=10,
            graph_count=10,  # fake the number of graphs as this is checked.
        )
        log._transient_data = {'graph_indices': [1, 2, 3]}
        yield log, i

  def RunMinibatch(self,
                   log: log_database.BatchLogMeta,
                   i: int,
                   print_context: typing.Any = None
                  ) -> classifier_base.ClassifierBase.MinibatchResults:
    """Fake mini-batch 'run'."""
    log.loss = 0
    return classifier_base.ClassifierBase.MinibatchResults(
        y_true_1hot=np.array([np.array([1], dtype=np.int32)]),
        y_pred_1hot=np.array([np.array([0], dtype=np.int32)]),
    )


def test_SaveModel_adds_row_to_checkpoints_table(
    tempdir2: pathlib.Path, graph_db: graph_database.Database,
    log_db: log_database.Database):
  """Test saving a model to file."""
  FLAGS.working_dir = tempdir2

  model = MockModel(graph_db, log_db)
  model.InitializeModel()
  model.global_training_step = 10
  model.SaveModel(validation_accuracy=.5)

  with model.log_db.Session() as session:
    assert session.query(log_database.ModelCheckpointMeta).count() == 1
    assert session.query(log_database.ModelCheckpoint).count() == 1


def test_LoadModel(tempdir2: pathlib.Path, graph_db: graph_database.Database,
                   log_db: log_database.Database):
  """Test loading a model from file."""
  FLAGS.working_dir = tempdir2

  model = MockModel(graph_db, log_db)
  model.InitializeModel()
  model.epoch_num = 2
  model.global_training_step = 10
  model.mock_data = 100
  model.SaveModel(validation_accuracy=.5)

  model.epoch_num = 0
  model.mock_data = 0
  model.global_training_step = 0
  model.LoadModel(run_id=model.run_id, epoch_num=2)
  assert model.epoch_num == 2
  assert model.global_training_step == 10
  assert model.mock_data == 100


def test_LoadModel_unknown_saved_model_flag(tempdir2: pathlib.Path,
                                            graph_db: graph_database.Database,
                                            log_db: log_database.Database):
  """Test that error is raised if saved model contains unknown flag."""
  FLAGS.working_dir = tempdir2
  model = MockModel(graph_db, log_db)
  model.InitializeModel()
  model.SaveModel(validation_accuracy=.5)

  with model.log_db.Session(commit=True) as session:
    # Insert a new "unknown" model flag.
    session.add(
        log_database.Parameter(
            run_id=model.run_id,
            type=log_database.ParameterType.MODEL_FLAG,
            parameter='a new flag',
            pickled_value=pickle.dumps(5),
        ))

  with pytest.raises(EnvironmentError) as e_ctx:
    model.LoadModel(run_id=model.run_id, epoch_num=model.epoch_num)

  # Check that the LoadModel() specifically complains about the new flag value.
  assert 'a new flag' in str(e_ctx.value)


def test_ModelFlagsToDict_subclass_model_name(tempdir2: pathlib.Path,
                                              graph_db: graph_database.Database,
                                              log_db: log_database.Database):
  """Test that model name uses subclass name, not the base class."""
  FLAGS.working_dir = tempdir2

  model = MockModel(graph_db, log_db)
  assert 'model' in model.ModelFlagsToDict()
  assert model.ModelFlagsToDict()['model'] == 'MockModel'


def test_Train(tempdir2: pathlib.Path, graph_db: graph_database.Database,
               log_db: log_database.Database):
  """Test that training terminates and bumps the epoch number."""
  FLAGS.working_dir = tempdir2

  model = MockModel(graph_db, log_db)
  model.InitializeModel()
  model.Train(num_epochs=1)
  assert model.best_epoch_num == 1


def test_Train_epoch_num(tempdir2: pathlib.Path,
                         graph_db: graph_database.Database,
                         log_db: log_database.Database):
  """Test that epoch_num has expected value."""
  FLAGS.working_dir = tempdir2

  model = MockModel(graph_db, log_db)
  model.InitializeModel()
  assert model.epoch_num == 0
  model.Train(num_epochs=1)
  assert model.epoch_num == 1
  model.Train(num_epochs=1)
  assert model.epoch_num == 2
  model.Train(num_epochs=2)
  assert model.epoch_num == 4


def test_Train_batch_log_count(tempdir2: pathlib.Path,
                               graph_db: graph_database.Database,
                               log_db: log_database.Database):
  """Test that training produces only batch logs for {val,test} runs."""
  FLAGS.working_dir = tempdir2

  model = MockModel(graph_db, log_db)
  model.InitializeModel()
  model.Train(num_epochs=1)
  with log_db.Session() as session:
    # 10 train + 10 val + 10 test logs
    assert session.query(log_database.BatchLogMeta).count() == 30
    # 10 val + 10 test logs
    assert session.query(log_database.BatchLog).count() == 20

    query = session.query(log_database.BatchLogMeta)
    query = query.filter(log_database.BatchLogMeta.type == 'test')
    for batch_log_meta in query:
      assert batch_log_meta.batch_log

    query = session.query(log_database.BatchLogMeta)
    query = query.filter(log_database.BatchLogMeta.type == 'train')
    for batch_log_meta in query:
      assert not batch_log_meta.batch_log

  model.Train(num_epochs=1)
  with log_db.Session() as session:
    # + 10 train + 10 train (no change to val acc so there's no new test logs)
    assert session.query(log_database.BatchLogMeta).count() == 50


def test_Train_keeps_a_single_checkpoint_and_set_of_batch_logs(
    tempdir2: pathlib.Path, graph_db: graph_database.Database,
    log_db: log_database.Database):
  """Check that only a single model checkpoint and set of detailed val logs
  are kept."""
  FLAGS.working_dir = tempdir2

  model = MockModel(graph_db, log_db)
  model.InitializeModel()
  model.Train(num_epochs=1)

  # Force the model to believe that it performed worse than it did so that when
  # we next call Train() it bumps the "best" accuracy.
  log_db.engine.execute(
      sql.update(log_database.ModelCheckpointMeta).values({
          'validation_accuracy':
          -1,
      }))
  assert model.best_epoch_validation_accuracy == -1  # Sanity check

  model.Train(num_epochs=1)
  # assert model.best_epoch_validation_accuracy == 1  # Sanity check

  with log_db.Session() as session:
    # There should still only be a single model checkpoint.
    assert session.query(log_database.ModelCheckpoint).count() == 1
    assert session.query(log_database.ModelCheckpointMeta).count() == 1

    # The "best" epoch is the new one.
    assert session.query(log_database.ModelCheckpointMeta.epoch).one()[0] == 2

    # 10 val + 10 test
    assert session.query(log_database.BatchLog).count() == 20
    # Check that the new batch logs replace the old ones.
    detailed_logs = session.query(log_database.BatchLogMeta)
    detailed_logs = detailed_logs.join(log_database.BatchLog)
    for log in detailed_logs:
      log.epoch == 2


def test_Test_creates_batch_logs(tempdir2: pathlib.Path,
                                 graph_db: graph_database.Database,
                                 log_db: log_database.Database):
  """Test that testing produces batch logs."""
  FLAGS.working_dir = tempdir2

  model = MockModel(graph_db, log_db)
  model.InitializeModel()
  model.RunEpoch(epoch_type='test')
  with log_db.Session() as session:
    assert session.query(log_database.BatchLogMeta).count() == 10
    assert session.query(log_database.BatchLog).count() == 10


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
