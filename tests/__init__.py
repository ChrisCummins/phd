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
from unittest import TestCase, skipIf

import numpy as np

import cldrive


class TestCldrive(TestCase):
    def test_run(self):
        kernel = '''\
__kernel void A(__global int* data) {
    int tid = get_global_id(0);
    data[tid] *= 2.0
}
'''
        outputs = cldrive.run(kernel, inputs=cldrive.Inputs.SEQ,
                              gsize=cldrive.NDRange(4,1,1),
                              lsize=cldrive.NDRange(1,1,1))

        self.assertTrue(np.array_equal(outputs, [[0,2,4,6]]))
