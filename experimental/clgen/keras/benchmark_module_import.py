"""Profile Keras module initialization.

Things I learned from this file:
 * Module imports are approx: Numpy 2 ms, Keras 26 ms, TensorFlow 4,153 ms.
"""
import sys

import pytest
from absl import app
from absl import flags


FLAGS = flags.FLAGS


def test_Numpy_import(benchmark):
  """Benchmark numpy module import."""

  def Benchmark():
    import numpy as np
    a = np.ndarray(1)
    del a

  benchmark(Benchmark)


def test_Keras_import(benchmark):
  """Benchmark keras module import."""

  def Benchmark():
    from keras import models
    m = models.Sequential()
    del m

  benchmark(Benchmark)


def test_Tensorflow_import(benchmark):
  """Benchmark tensorflow module import."""

  def Benchmark():
    import tensorflow as tf
    a = tf.Variable(1)
    del a

  benchmark(Benchmark)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments '{}'".format(', '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
