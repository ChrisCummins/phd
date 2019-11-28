"""Smoke test for //deeplearning/ml4pl/models/ggnn."""
import numpy as np

from deeplearning.ml4pl.models import classifier_smoke_tester
from deeplearning.ml4pl.models.ggnn import ggnn
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

MODULE_UNDER_TEST = "deeplearning.ml4pl.models.ggnn"

FLAGS = app.FLAGS


def test_binary_node_labels():
  """Test classification with binary node labels."""
  classifier_smoke_tester.RunSmokeTest(
    ggnn.GgnnClassifier,
    node_y_choices=[
      np.array([1, 0], dtype=np.int32),
      np.array([0, 1], dtype=np.int32),
    ],
  )


def test_graph_features_and_labels():
  """Test classification with graph-level features and labels."""
  classifier_smoke_tester.RunSmokeTest(
    ggnn.GgnnClassifier,
    graph_x_choices=[
      np.array([32, 64], dtype=np.int32),
      np.array([128, 1024], dtype=np.int32),
      np.array([256, 13], dtype=np.int32),
    ],
    graph_y_choices=[
      np.array([1, 0], dtype=np.int32),
      np.array([0, 1], dtype=np.int32),
    ],
  )


if __name__ == "__main__":
  test.Main(capture_output=False)
