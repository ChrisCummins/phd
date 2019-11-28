"""Unit tests for //deeplearning/ml4pl/models:log_database."""
import pathlib
import pickle

import numpy as np
import pytest

from deeplearning.ml4pl.models import log_database
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def db(tempdir: pathlib.Path):
  return log_database.Database(f'sqlite:///{tempdir}/db')


def MakeBatchLog():
  log = log_database.BatchLogMeta(run_id='20191023@foo',
                                  epoch=10,
                                  batch=0,
                                  global_step=1024,
                                  elapsed_time_seconds=.5,
                                  graph_count=100,
                                  node_count=500,
                                  loss=.25,
                                  precision=.5,
                                  recall=.5,
                                  f1=.5,
                                  accuracy=.75,
                                  type="train",
                                  group="0",
                                  batch_log=log_database.BatchLog())
  log.graph_indices = [0, 1, 2, 3]
  log.predictions = np.array([0, 1, 2, 3])
  log.accuracies = np.array([True, False, False])
  return log


def test_BatchLogMeta_columns(db: log_database.Database):
  with db.Session(commit=True) as session:
    session.add(MakeBatchLog())

  with db.Session() as session:
    log = session.query(log_database.BatchLogMeta).first()
    assert log.run_id == '20191023@foo'
    assert log.epoch == 10
    assert log.batch == 0
    assert log.global_step == 1024
    assert log.elapsed_time_seconds == .5
    assert log.graph_count == 100
    assert log.node_count == 500
    assert log.loss == .25
    assert log.precision == .5
    assert log.recall == .5
    assert log.f1 == .5
    assert log.accuracy == .75
    assert log.type == "train"
    assert log.group == "0"
    assert log.graph_indices == [0, 1, 2, 3]
    assert np.array_equal(log.predictions, np.array([0, 1, 2, 3]))
    assert np.array_equal(log.accuracies, np.array([True, False, False]))


def test_DeleteLogsForRunId(db: log_database.Database):
  """Test that delete batch log meta cascades to batch log."""
  with db.Session(commit=True) as session:
    log = MakeBatchLog()
    run_id = log.run_id
    session.add(log)

    session.add(
        log_database.Parameter(run_id=run_id,
                               parameter='foo',
                               type=log_database.ParameterType.MODEL_FLAG,
                               pickled_value=pickle.dumps('foo')))

  db.DeleteLogsForRunId(run_id)

  with db.Session() as session:
    assert not session.query(log_database.BatchLogMeta.id).count()
    assert not session.query(log_database.BatchLog.id).count()
    assert not session.query(log_database.Parameter.id).count()


def test_run_ids(db: log_database.Database):
  """Test that property returns all run IDs."""
  with db.Session(commit=True) as session:
    session.add_all([
        log_database.Parameter(run_id='a',
                               type=log_database.ParameterType.MODEL_FLAG,
                               parameter='foo',
                               pickled_value=pickle.dumps('foo')),
        log_database.Parameter(run_id='a',
                               type=log_database.ParameterType.MODEL_FLAG,
                               parameter='bar',
                               pickled_value=pickle.dumps('bar')),
        log_database.Parameter(run_id='b',
                               type=log_database.ParameterType.MODEL_FLAG,
                               parameter='foo',
                               pickled_value=pickle.dumps('foo')),
    ])
  assert db.run_ids == ['a', 'b']


if __name__ == '__main__':
  test.Main()
