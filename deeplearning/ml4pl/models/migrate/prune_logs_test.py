"""Unit tests for //deeplearning/ml4pl/models/migrate:prune_logs."""
import pathlib
import pickle

import pytest

from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models.migrate import prune_logs
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def db(tempdir: pathlib.Path):
  return log_database.Database(f'sqlite:///{tempdir}/db')


@pytest.fixture(scope='function')
def input_db(db: log_database.Database) -> log_database.Database:
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
        log_database.ModelCheckpointMeta(
            run_id='a',
            epoch=0,
            global_step=0,
            validation_accuracy=.5,
            model_checkpoint=log_database.ModelCheckpoint(
                pickled_data=pickle.dumps('checkpoint'))),
        log_database.BatchLogMeta(
            run_id='a',
            epoch=0,
            batch=1,
            type='train',
            group='train',
            global_step=0,
            elapsed_time_seconds=.5,
            graph_count=1,
            node_count=100,
            loss=.5,
            accuracy=.5,
            precision=.5,
            recall=.5,
            f1=.5,
            batch_log=log_database.BatchLog(
                pickled_graph_indices=pickle.dumps([1, 2, 3]),
                pickled_accuracies=pickle.dumps([1, 2, 3]),
                pickled_predictions=pickle.dumps([1, 2, 3]))),
        log_database.BatchLogMeta(
            run_id='b',
            epoch=0,
            batch=1,
            type='train',
            group='train',
            global_step=0,
            elapsed_time_seconds=.5,
            graph_count=1,
            node_count=100,
            loss=.5,
            accuracy=.5,
            precision=.5,
            recall=.5,
            f1=.5,
            batch_log=log_database.BatchLog(
                pickled_graph_indices=pickle.dumps([1, 2, 3]),
                pickled_accuracies=pickle.dumps([1, 2, 3]),
                pickled_predictions=pickle.dumps([1, 2, 3]))),
    ])
  return db


def test_PruneRunsWithNoCheckpoints(input_db: log_database.Database):
  prune_logs.PruneRunsWithNoCheckpoints(input_db)
  with input_db.Session() as session:
    assert session.query(
        log_database.Parameter.run_id.distinct()).all() == [('a',)]
    assert session.query(
        log_database.ModelCheckpointMeta.run_id.distinct()).all() == [('a',)]
    assert session.query(
        log_database.BatchLogMeta.run_id.distinct()).all() == [('a',)]


if __name__ == '__main__':
  test.Main()
