"""Unit tests for //docs:minimize_bibtex."""
from docs import minimize_bibtex

from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


class MockBibtex(object):
  """Mock class for bibtex."""

  def __init__(self):
    self.entries = []


def test_DeleteKeysInPlace():
  dictionary = {
      'a': 1,
      'b': 2,
      'c': 3,
  }
  minimize_bibtex.DeleteKeysInPlace(dictionary, ['a', 'b', 'd'])

  assert dictionary == {'c': 3}


def test_MinimizeBibtexInPlace():
  bibtex = MockBibtex()
  bibtex.entries.append({
      'a': 1,
      'abstract': 2,
      'url': 3,
  })
  bibtex.entries.append({
      'b': 1,
      'c': 2,
      'd': 3,
  })

  minimize_bibtex.MinimizeBibtexInPlace(bibtex)

  assert bibtex.entries == [{
      'a': 1
  }, {
      'b': 1,
      'c': 2,
      'd': 3,
  }]


if __name__ == '__main__':
  test.Main()
