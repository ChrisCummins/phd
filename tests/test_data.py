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
from unittest import TestCase, skip, main

import numpy as np

from numpy import testing as nptest

from tests import *

import cldrive


class TestData(TestCase):
    def test_zeros(self):
        src = """ kernel void A(global float* a) {} """

        outputs = cldrive.zeros(src, 64)
        outputs_gs = [np.zeros(64)]

        almost_equal(outputs, outputs_gs)

    def test_ones(self):
        src = """ kernel void A(global float* a, const int b) {} """

        outputs = cldrive.ones(src, 1024)
        outputs_gs = [np.ones(1024), [1024]]

        almost_equal(outputs, outputs_gs)

    def test_arange(self):
        src = "kernel void A(global float* a, local float* b, const int c) {}"

        outputs = cldrive.arange(src, 512, scalar_val=0)
        outputs_gs = [np.arange(512), [0]]

        almost_equal(outputs, outputs_gs)

    def test_rand(self):
        src = "kernel void A(global float* a, global float* b) {}"

        outputs = cldrive.rand(src, 16, scalar_val=0)

        # we can't test the actual values
        self.assertEqual(outputs.shape, (2, 16))


if __name__ == "__main__":
    main()
