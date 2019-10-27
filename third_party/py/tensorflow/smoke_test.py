"""Test that tensorflow can be imported."""
import sys

import numpy as np
import pytest
from labm8 import app
from labm8 import test

import getconfig

FLAGS = app.FLAGS

MODULE_UNDER_TEST = None  # No test coverage.


def test_import_tensorflow():
  print('Python executable:', sys.executable)
  print('Python version:', sys.version)
  import tensorflow
  print('Tensorflow:', tensorflow.__file__)
  print('Tensorflow version:', tensorflow.VERSION)


def test_tensorflow_session():
  import tensorflow as tf
  a = tf.constant(1)
  b = tf.constant(2)
  c = a + b
  with tf.compat.v1.Session() as sess:
    assert sess.run(c) == 3


# If the project has been configured to use CUDA, this test will pin an
# operation to the GPU and test that it works.
@pytest.mark.skipif(not getconfig.GetGlobalConfig().with_cuda, reason='No GPU')
def test_tensorflow_gpu_constant():
  import tensorflow as tf
  with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
      log_device_placement=True)) as sess:
    with tf.device('/gpu:0'):
      assert sess.run(tf.constant(1)) == 1


# If the project has been configured to use CUDA, this test will pin an
# operation to the GPU and test that it works.
@pytest.mark.skipif(not getconfig.GetGlobalConfig().with_cuda, reason='No GPU')
def test_tensorflow_gpu_computation():
  import tensorflow as tf
  with tf.device('/gpu:0'):
    a = tf.constant([1, 2, 3, 4, 5, 6],
                    shape=[2, 3],
                    name='a',
                    dtype=tf.float32)
    b = tf.constant([1, 2, 3, 4, 5, 6],
                    shape=[3, 2],
                    name='b',
                    dtype=tf.float32)
    c = tf.matmul(a, b)
  with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(
      log_device_placement=True)) as sess:
    np.testing.assert_array_almost_equal(sess.run(c),
                                         np.array([[22, 28], [49, 64]]))


if __name__ == '__main__':
  test.Main()
