"""Smoke test for //deeplearning/ml4pl/models/zero_r."""
from labm8 import app

from deeplearning.ml4pl.models import classifier_smoke_tester
from deeplearning.ml4pl.models.zero_r import zero_r

FLAGS = app.FLAGS


def main():
  classifier_smoke_tester.RunNodeClassificationSmokeTest(zero_r.ZeroRClassifier)


if __name__ == '__main__':
  app.Run(main)
