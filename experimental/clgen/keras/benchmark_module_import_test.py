"""Profile Keras module initialization.

Things I learned from this file:
 * Module imports are approx: Numpy 2 ms, Keras 26 ms, TensorFlow 4,153 ms.
"""

from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


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


if __name__ == '__main__':
  test.Main()
