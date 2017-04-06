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

from tests import *

import cldrive


class TestDriver(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestDriver, self).__init__(*args, **kwargs)

    def test_simple(self):
        inputs      = [[0, 1, 2, 3, 4,  5,  6,  7]]
        inputs_orig = [[0, 1, 2, 3, 4,  5,  6,  7]]
        outputs_gs  = [[0, 2, 4, 6, 8, 10, 12, 14]]

        src = """
        kernel void A(global float* a) {
            const int x_id = get_global_id(0);

            a[x_id] *= 2.0;
        }
        """

        driver = cldrive.Driver(ENV, src)
        outputs = driver(inputs, gsize=(8, 1, 1), lsize=(1, 1, 1))

        # inputs are unmodified
        almost_equal(inputs, inputs_orig)
        # outputs
        almost_equal(outputs, outputs_gs)

    def test_double_inputs(self):
        inputs      = [[0, 1, 2, 3, 0, 1, 2, 3],  [2, 4]]
        inputs_orig = [[0, 1, 2, 3, 0, 1, 2, 3],  [2, 4]]
        outputs_gs  = [[0, 2, 4, 6, 0, 4, 8, 12], [2, 4]]

        src = """
        kernel void A(global int* a, const int2 b) {
            const int x_id = get_global_id(0);
            const int y_id = get_global_id(1);

            if (!y_id) {
                a[x_id] *= b.x;
            } else {
                a[get_global_size(0) + x_id] *= b.y;
            }
        }
        """

        driver = cldrive.Driver(ENV, src)
        outputs = driver(inputs, gsize=(4, 2, 1), lsize=(1, 1, 1))

        almost_equal(inputs, inputs_orig)
        almost_equal(outputs, outputs_gs)

        # run kernel a second time with the previous outputs
        outputs2 = driver(outputs, gsize=(4, 2, 1), lsize=(1, 1, 1))
        outputs2_gs  = [[0, 4, 8, 12, 0, 16, 32, 48], [2, 4]]
        almost_equal(outputs2, outputs2_gs)


# TODO: Difftest against cl_launcher from CLSmith for a CLSmith kernel.


if __name__ == "__main__":
    main()
