"""Unit tests for //deeplearning/ml4pl/models:log_analysis."""
import pathlib
import random

import numpy as np
import pytest

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.models import log_analysis
from deeplearning.ml4pl.models import log_database
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def graph_db(tempdir: pathlib.Path) -> graph_database.Database:
  return graph_database.Database(f'sqlite:///{tempdir}/graphs.db')


@pytest.fixture(scope='function')
def db(tempdir: pathlib.Path) -> log_database.Database:
  return log_database.Database(f'sqlite:///{tempdir}/db')


def GenerateRandomBatchLogs(run_id: str, num_epochs: int = 3):
  """Generate batch logs with fake data."""
  global_step = 0
  for epoch_num in range(num_epochs):
    for epoch_type in ['train', 'val', 'test']:
      for batch_num in range(100):
        log = log_database.BatchLogMeta(run_id=run_id,
                                        epoch=epoch_num,
                                        batch=batch_num,
                                        global_step=global_step,
                                        elapsed_time_seconds=random.random(),
                                        graph_count=random.randint(10, 50),
                                        node_count=random.randint(100, 1000),
                                        loss=random.random(),
                                        precision=random.random(),
                                        recall=random.random(),
                                        f1=random.random(),
                                        accuracy=random.random(),
                                        type=epoch_type,
                                        group=epoch_type,
                                        batch_log=log_database.BatchLog())
        log.graph_indices = [0, 1, 2, 3]
        log.predictions = np.array([0, 1, 2, 3])
        log.accuracies = np.array([True, False, False])
        yield log
        global_step += 1


def test_RunLogAnalyzer_epoch_logs(graph_db: graph_database.Database,
                                   db: log_database.Database):
  with db.Session(commit=True) as session:
    session.add_all(list(GenerateRandomBatchLogs('foo')))
    session.add_all(list(GenerateRandomBatchLogs('bar')))
  run = log_analysis.RunLogAnalyzer(graph_db, db, 'foo')

  assert len(run.epoch_logs) == 3 * 3  # epoch_count * type


def test_RunLogAnalyzer_batch_logs(graph_db: graph_database.Database,
                                   db: log_database.Database):
  with db.Session(commit=True) as session:
    session.add_all(list(GenerateRandomBatchLogs('foo')))
    session.add_all(list(GenerateRandomBatchLogs('bar')))
  run = log_analysis.RunLogAnalyzer(graph_db, db, 'foo')

  assert len(run.batch_logs) == 3 * 3 * 100  # epoch_count * type * num_batches


def test_BuildConfusionMatrix():
  confusion_matrix = log_analysis.BuildConfusionMatrix(
      targets=np.array([
          np.array([1, 0, 0], dtype=np.int32),
          np.array([0, 0, 1], dtype=np.int32),
          np.array([0, 0, 1], dtype=np.int32),
      ]),
      predictions=np.array([
          np.array([.1, 0.5, 0], dtype=np.float32),
          np.array([0, -.5, -.3], dtype=np.float32),
          np.array([0, 0, .8], dtype=np.float32),
      ]))

  assert confusion_matrix.shape == (3, 3)
  assert confusion_matrix.sum() == 3
  assert np.array_equal(confusion_matrix,
                        np.array([
                            [0, 1, 0],
                            [0, 0, 0],
                            [1, 0, 1],
                        ]))


if __name__ == '__main__':
  test.Main()
