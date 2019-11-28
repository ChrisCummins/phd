"""Tests for //learn/tensorflow:mnist_regression."""
from labm8.py import app
from labm8.py import test
from learn.tensorflow import mnist_regression

FLAGS = app.FLAGS


def test_HyperParamSweep_smoke_test():
  """Test that hyper param sweep doesn't blow up."""
  assert mnist_regression.HyperParamSweep(1)


if __name__ == "__main__":
  test.Main()
