"""Unit tests for //deeplearning/ml4pl/models/lstm:graph2seq."""
import pathlib

import pytest
from labm8 import app
from labm8 import test

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.models.lstm import graph2seq

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def bytecode_db(tempdir: pathlib.Path) -> bytecode_database.Database:
  yield bytecode_database.Database(f'sqlite:///{tempdir}/bytecodes.db')


@pytest.fixture(scope='function')
def graph_db(tempdir: pathlib.Path) -> graph_database.Database:
  yield graph_database.Database(f'sqlite:///{tempdir}/graphs.db')


def test_Encode(bytecode_db: bytecode_database.Database,
                graph_db: graph_database.Database):
  FLAGS.bytecode_db = bytecode_db.url

  encoder = graph2seq.GraphToSequenceEncoder(graph_db)
  encoded = encoder.Encode(["Hello, world"])
  assert len(encoded) == 1
  assert encoded[0]


if __name__ == '__main__':
  test.Main()
