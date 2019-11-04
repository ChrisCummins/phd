"""Smoke test for //deeplearning/ml4pl/models/ggnn:ggnn_node_classifier."""
import numpy as np
from labm8 import app

from deeplearning.ml4pl.models import classifier_smoke_tester
from deeplearning.ml4pl.models.ggnn import ggnn_node_classifier

FLAGS = app.FLAGS


def main():
  # Binary node labels.
  classifier_smoke_tester.RunSmokeTest(
      ggnn_node_classifier.GgnnNodeClassifierModel,
      node_y_choices=[
          np.array([1, 0], dtype=np.int32),
          np.array([0, 1], dtype=np.int32),
      ])

  # Graph features and labels.
  classifier_smoke_tester.RunSmokeTest(
      ggnn_node_classifier.GgnnNodeClassifierModel,
      graph_x_choices=[
          np.array([32, 64], dtype=np.int32),
          np.array([128, 1024], dtype=np.int32),
          np.array([256, 13], dtype=np.int32),
      ],
      graph_y_choices=[
          np.array([1, 0], dtype=np.int32),
          np.array([0, 1], dtype=np.int32),
      ])


if __name__ == '__main__':
  app.Run(main)
