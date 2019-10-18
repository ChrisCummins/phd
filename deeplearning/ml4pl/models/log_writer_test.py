"""Unit tests for //deeplearning/ml4pl/models:log_writer."""
import time

import pathlib

from deeplearning.ml4pl.models import log_writer
from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def test_FormattedJsonLogWriter(tempdir: pathlib.Path):
  writer = log_writer.FormattedJsonLogWriter(tempdir)

  assert len(list(writer.Logs())) == 0
  writer.Log({'a': 1})
  assert len(list(writer.Logs())) == 1
  time.sleep(.1)
  writer.Log({'b': 2})
  print(list(writer.outpath.iterdir()))
  assert len(list(writer.Logs())) == 2


if __name__ == '__main__':
  test.Main()
