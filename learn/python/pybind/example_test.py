"""Tests for pybind python extension."""
from labm8.py import test
from learn.python.pybind import example

FLAGS = test.FLAGS


def test_add():
  """Test calling into C++ bound extension."""
  assert example.add(1, 2) == 3


if __name__ == "__main__":
  test.Main()
