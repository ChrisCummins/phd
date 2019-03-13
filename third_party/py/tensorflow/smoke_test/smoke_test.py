"""Test that tensorflow can be imported."""

import sys

from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def test_import_tensorflow():
  print('Python executable:', sys.executable)
  print('Python version:', sys.version)
  import tensorflow
  print('Tensorflow:', tensorflow.__file__)
  print('Tensorflow version:', tensorflow.VERSION)


if __name__ == '__main__':
  test.Main()
