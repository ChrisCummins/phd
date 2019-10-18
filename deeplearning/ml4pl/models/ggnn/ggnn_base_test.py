"""Unit tests for //deeplearning/ml4pl/graphs/labelled/reachability."""

from deeplearning.ml4pl.models.ggnn import ggnn_base
from labm8 import app
from labm8 import test


FLAGS = app.FLAGS

def test_FlagsToDict():
  d = ggnn_base.FlagsToDict()
  print(d)
  assert not d


if __name__ == '__main__':
  test.Main()
