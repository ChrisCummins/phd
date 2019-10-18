"""Unit tests for //deeplearning/ml4pl/models:log_writer."""
import pathlib

from deeplearning.ml4pl.models import batch_logger
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def test_InMemoryBatchLogger(tempdir: pathlib.Path):
  logger = batch_logger.InMemoryBatchLogger(tempdir)

  logger.Log(batch_size=10, loss=0.5, accuracy=0.5)
  assert logger.batch_count == 1
  assert logger.average_batch_size == 10
  assert logger.average_loss == 0.5
  assert logger.average_accuracy == 0.5
  assert logger.instances_per_second > 0

  logger.Log(batch_size=20, loss=0.0, accuracy=1.0)
  assert logger.batch_count == 2
  assert logger.average_batch_size == 15
  assert logger.average_loss == 0.25
  assert logger.average_accuracy == 0.75
  assert logger.instances_per_second > 0


if __name__ == '__main__':
  test.Main()
