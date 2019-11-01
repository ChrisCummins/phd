"""Unit tests for //deeplearning/ml4pl/models:log_database."""
import pathlib

import numpy as np
import pytest
from labm8 import app
from labm8 import test

from deeplearning.ml4pl.models import log_database

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def db(tempdir: pathlib.Path):
  return log_database.Database(f'sqlite:///{tempdir}/db')


def test_BatchLog_columns(db: log_database.Database):
  with db.Session(commit=True) as session:
    log = log_database.BatchLog(run_id='20191023@foo',
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
                                group="train",
                                is_training=True,
                                instances=log_database.Instances())
    log.graph_indices = [0, 1, 2, 3]
    log.predictions = np.array([0, 1, 2, 3])
    log.accuracies = np.array([True, False, False])
    session.add(log)

  with db.Session() as session:
    log = session.query(log_database.BatchLog).first()
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
    assert log.group == "train"
    assert log.graph_indices == [0, 1, 2, 3]
    assert np.array_equal(log.predictions, np.array([0, 1, 2, 3]))
    assert np.array_equal(log.accuracies, np.array([True, False, False]))


if __name__ == '__main__':
  test.Main()
