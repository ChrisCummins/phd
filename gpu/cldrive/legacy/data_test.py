"""Unit tests for //gpu/cldrive/legacy/data.py."""

import numpy as np
from absl import flags

from gpu.cldrive.legacy import data
from gpu.cldrive.legacy import testlib
from labm8 import test

FLAGS = flags.FLAGS


def test_MakeZeros():
  """Basic test for data generator."""
  outputs = data.MakeZeros("kernel void A(global float* a) {}", 64)
  outputs_gs = [np.zeros(64)]
  testlib.Assert2DArraysAlmostEqual(outputs, outputs_gs)


def test_MakeOnes():
  """Basic test for data generator."""
  outputs = data.MakeOnes("kernel void A(global float* a, const int b) {}",
                          1024)
  outputs_gs = [np.ones(1024), [1024]]
  testlib.Assert2DArraysAlmostEqual(outputs, outputs_gs)


def test_MakeArange():
  """Basic test for data generator."""
  outputs = data.MakeArange(
      "kernel void A(global float* a, local float* b, const int c) {}",
      512,
      scalar_val=0)
  outputs_gs = [np.arange(512), [0]]
  testlib.Assert2DArraysAlmostEqual(outputs, outputs_gs)


def test_MakeRand():
  """Basic test for data generator."""
  outputs = data.MakeRand("kernel void A(global float* a, global float* b) {}",
                          16)
  assert outputs.shape == (2, 16)


if __name__ == "__main__":
  test.Main()
