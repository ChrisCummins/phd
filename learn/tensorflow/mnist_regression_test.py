"""Tests for //learn/tensorflow:mnist_regression."""
from absl import flags

from labm8 import test
from learn.tensorflow import mnist_regression

FLAGS = flags.FLAGS


def test_HyperParamSweep_smoke_test():
  """Test that hyper param sweep doesn't blow up."""
  assert mnist_regression.HyperParamSweep(1)


if __name__ == '__main__':
  test.Main()
