"""Unit tests for //deeplearning/ml4pl/models:log_analysis."""
import numpy as np
from labm8 import app
from labm8 import test

from deeplearning.ml4pl.models import log_analysis

FLAGS = app.FLAGS


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
