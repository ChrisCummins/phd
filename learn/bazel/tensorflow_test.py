"""A simple test to ensure that TensorFlow is working."""

import tensorflow as tf
from absl import flags

from labm8 import test

FLAGS = flags.FLAGS


def test_Tensorflow_hello_world():
  """The "hello world" of TensorFlow tests."""
  sess = tf.Session()
  a = tf.constant(10)
  b = tf.constant(32)
  assert 42 == sess.run(a + b)
  sess.close()


if __name__ == '__main__':
  test.Main()
