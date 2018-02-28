# Copyright (C) 2017 Chris Cummins.
#
# This file is part of cldrive.
#
# Cldrive is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Cldrive is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <http://www.gnu.org/licenses/>.
#
import pytest

import numpy as np
from numpy import testing as nptest

import cldrive

from lib import *


def test_zeros():
    src = "kernel void A(global float* a) {}"

    outputs = cldrive.zeros(src, 64)
    outputs_gs = [np.zeros(64)]

    almost_equal(outputs, outputs_gs)


def test_ones():
    src = "kernel void A(global float* a, const int b) {}"

    outputs = cldrive.ones(src, 1024)
    outputs_gs = [np.ones(1024), [1024]]

    almost_equal(outputs, outputs_gs)


def test_arange():
    src = "kernel void A(global float* a, local float* b, const int c) {}"

    outputs = cldrive.arange(src, 512, scalar_val=0)
    outputs_gs = [np.arange(512), [0]]

    almost_equal(outputs, outputs_gs)


def test_rand():
    src = "kernel void A(global float* a, global float* b) {}"

    outputs = cldrive.rand(src, 16)

    # we can't test the actual values
    assert outputs.shape == (2, 16)


@skip_on_pocl
def test_data_unchanged():
    src = "kernel void A(global int* a, global int* b, const int c) {}"

    inputs = cldrive.rand(src, 16)
    outputs = cldrive.drive(ENV, src, inputs, gsize=(16,1,1), lsize=(1,1,1))

    almost_equal(outputs, inputs)


@skip_on_pocl
def test_data_zerod():
    # zero-ing a randomly initialized array
    src = "kernel void A(global int* a) { a[get_global_id(0)] = 0; }"

    inputs = cldrive.rand(src, 16)
    outputs = cldrive.drive(ENV, src, inputs, gsize=(16,1,1), lsize=(4,1,1))

    almost_equal(outputs, [np.zeros(16)])


@skip_on_pocl
def test_vector_input_switch():
    src = """
    kernel void A(global int2* a) {
        const int tid = get_global_id(0);

        const int tmp = a[tid].x;
        a[tid].x = a[tid].y;
        a[tid].y = tmp;
    }
    """

    inputs = cldrive.arange(src, 4)
    outputs_gs = [[1, 0, 3, 2, 5, 4, 7, 6]]

    outputs = cldrive.drive(ENV, src, inputs, gsize=(4,1,1), lsize=(4,1,1))

    almost_equal(outputs, outputs_gs)
