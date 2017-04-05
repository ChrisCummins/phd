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
    def test_extract_args(self):
        src = """
    typedef int foobar;

    void B(const int e);

    __kernel void A(const __global int* data, __local float4 * restrict car,
                    __global const float* b, const int foo, int d) {
        int tid = get_global_id(0);
        data[tid] *= 2.0;
    }

    void B(const int e) {}
    """
        args = cldrive.extract_args(src)
        self.assertEqual(len(args), 5)

    def test_extract_args_multiple_kernels(self):
        src = """
    __kernel void A(__global int* a) {}
    __kernel void B(const int e) {}
    """
        with self.assertRaises(cldrive.ParseError):
            cldrive.extract_args(src)

    def test_extract_args_no_args(self):
        src = """__kernel void A() {}"""
        args = cldrive.extract_args(src)
        self.assertEqual(len(args), 0)

    def test_extract_args_no_kernel(self):
        src = """__kernel void A();"""
        with self.assertRaises(cldrive.ParseError):
            cldrive.extract_args(src)

    def test_make_env_not_found(self):
        with self.assertRaises(cldrive.OpenCLDeviceNotFound):
            cldrive.make_env(platform_id=9999999, device_id=9999999)

    def test_run(self):
        kernel = '''\
__kernel void A(__global int* data) {
    int tid = get_global_id(0);
    data[tid] *= 2.0;
}
'''
        env = cldrive.make_env()

        outputs = cldrive.run_kernel(kernel, data_generator=cldrive.Generator.SEQ,
                                     gsize=cldrive.NDRange(4,1,1),
                                     lsize=cldrive.NDRange(1,1,1),
                                     env=env)
        self.assertTrue(np.array_equal(outputs, [[0,2,4,6]]))

        outputs = cldrive.run_kernel(kernel, data_generator=cldrive.Generator.SEQ,
                                     gsize=cldrive.NDRange(8,1,1),
                                     lsize=cldrive.NDRange(1,1,1),
                                     env=env)
        # self.assertTrue(np.array_equal(outputs, [[0,2,4,6,8,10,12,14]]))
