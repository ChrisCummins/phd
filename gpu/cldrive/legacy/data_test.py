# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
# This file is part of cldrive.
#
# cldrive is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cldrive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //gpu/cldrive/legacy/data.py."""
import numpy as np

from gpu.cldrive.legacy import data
from gpu.cldrive.legacy import testlib
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


def test_MakeZeros():
  """Basic test for data generator."""
  outputs = data.MakeZeros("kernel void A(global float* a) {}", 64)
  outputs_gs = [np.zeros(64)]
  testlib.Assert2DArraysAlmostEqual(outputs, outputs_gs)


def test_MakeOnes():
  """Basic test for data generator."""
  outputs = data.MakeOnes(
    "kernel void A(global float* a, const int b) {}", 1024
  )
  outputs_gs = [np.ones(1024), [1024]]
  testlib.Assert2DArraysAlmostEqual(outputs, outputs_gs)


def test_MakeArange():
  """Basic test for data generator."""
  outputs = data.MakeArange(
    "kernel void A(global float* a, local float* b, const int c) {}",
    512,
    scalar_val=0,
  )
  outputs_gs = [np.arange(512), [0]]
  testlib.Assert2DArraysAlmostEqual(outputs, outputs_gs)


def test_MakeRand():
  """Basic test for data generator."""
  outputs = data.MakeRand(
    "kernel void A(global float* a, global float* b) {}", 16
  )
  assert outputs.shape == (2, 16)


if __name__ == "__main__":
  test.Main()
