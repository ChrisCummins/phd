"""A simple test to ensure that TensorFlow is working."""
import tensorflow as tf

from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

MODULE_UNDER_TEST = None  # No coverage.


def test_Tensorflow_hello_world():
  """The "hello world" of TensorFlow tests."""
  sess = tf.compat.v1.Session()
  a = tf.constant(10)
  b = tf.constant(32)
  assert 42 == sess.run(a + b)
  sess.close()


if __name__ == "__main__":
  test.Main()
