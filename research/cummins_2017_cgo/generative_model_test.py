"""Unit tests for //research/cummins_2017_cgo:generative_model."""
from labm8 import test

from research.cummins_2017_cgo import generative_model


def test_CreateInstanceProtoFromFlags_smoke_test():
  """Test that instance proto can be constructed."""
  assert generative_model.CreateInstanceProtoFromFlags()


def test_CreateInstanceFromFlags_smoke_test():
  """Test that instance can be constructed."""
  assert generative_model.CreateInstanceFromFlags()


if __name__ == '__main__':
  test.Main()
