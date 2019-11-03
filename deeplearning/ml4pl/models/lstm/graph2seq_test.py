"""Unit tests for //deeplearning/ml4pl/models/lstm:graph2seq."""
import pathlib

import numpy as np
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


def test_Encode(graph_db: graph_database.Database):
  encoder = graph2seq.GraphToSequenceEncoder(graph_db)
  encoded = encoder.Encode(["Hello, world"])
  assert len(encoded) == 1
  assert encoded[0] == [124, 7, 6, 6, 3, 51, 1, 71, 80, 6, 4]


def test_StringsToEncodedSequencesAndGroupings():
  encoder = graph2seq.GraphToSequenceEncoder(graph_db)

  enc, idx = encoder.StringsToEncodedSequencesAndGroupings(["HH", "H", "H"])

  assert len(enc) == 4
  assert np.array_equal(enc, [124, 124, 124, 124])
  assert np.array_equal(idx, [0, 0, 1, 2])


if __name__ == '__main__':
  test.Main()
