"""Smoke test for //deeplearning/ml4pl/models/zero_r."""
import numpy as np
from labm8 import app

from deeplearning.ml4pl.models import classifier_smoke_tester
from deeplearning.ml4pl.models.zero_r import zero_r

FLAGS = app.FLAGS


def main():
  classifier_smoke_tester.RunSmokeTest(zero_r.ZeroRClassifier,
                                       node_y_choices=[
                                           np.array([1, 0], dtype=np.int32),
                                           np.array([0, 1], dtype=np.int32),
                                       ])


if __name__ == '__main__':
  app.Run(main)
