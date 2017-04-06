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

        driver = cldrive.Driver(ENV, src)
        outputs = cldrive.zeros(driver, 64)
        outputs_gs = [np.zeros(64)]

        almost_equal(outputs, outputs_gs)

    def test_ones(self):
        src = """ kernel void A(global float* a, const int b) {} """

        driver = cldrive.Driver(ENV, src)
        outputs = cldrive.ones(driver, 1024)
        outputs_gs = [np.ones(1024), [1024]]

        almost_equal(outputs, outputs_gs)

    def test_arange(self):
        src = "kernel void A(global float* a, local float* b, const int c) {}"

        driver = cldrive.Driver(ENV, src)
        outputs = cldrive.arange(driver, 512, scalar_val=0)
        outputs_gs = [np.arange(512), [0]]

        almost_equal(outputs, outputs_gs)

    def test_rand(self):
        src = "kernel void A(global float* a, global float* b) {}"

        driver = cldrive.Driver(ENV, src)
        outputs = cldrive.rand(driver, 16, scalar_val=0)

        # we can't test the actual values
        self.assertEqual(outputs.shape, (2, 16))


if __name__ == "__main__":
    main()
