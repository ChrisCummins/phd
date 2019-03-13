"""Test that tensorflow can be imported."""

from labm8 import app
from labm8 import test

FLAGS = app.FLAGS


def test_import_tensorflow():
  import sys
  print('Python executable:', sys.executable)
  print('Python version:', sys.version)
  import tensorflow
  print('Tensorflow version:', tensorflow.__version__)


if __name__ == '__main__':
  test.Main()
