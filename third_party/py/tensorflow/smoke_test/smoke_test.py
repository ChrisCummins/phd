"""Test that tensorflow can be imported."""
import pytest
import sys

from config import getconfig
from labm8 import app
from labm8 import test

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
  with tf.Session() as sess:
    assert sess.run(c) == 3


# If the project has been configured to use CUDA, this test will pin an
# operation to the GPU and test that it works.
@pytest.mark.skipif(not getconfig.GetGlobalConfig().with_cuda, reason='No GPU')
def test_tensorflow_on_gpu():
  import tensorflow as tf
  with tf.device('/cpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
    c = tf.matmul(a, b)
  with tf.Session() as sess:
    sess.run(c)


if __name__ == '__main__':
  test.Main()
