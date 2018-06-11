"""A simple test to ensure that TensorFlow is working."""
import sys

import pytest
import tensorflow as tf
from absl import app
from absl import flags


def test_Tensorflow_hello_world():
  """The "hello world" of TensorFlow tests."""
  sess = tf.Session()
  a = tf.constant(10)
  b = tf.constant(32)
  assert 42 == sess.run(a + b)
  sess.close()


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
