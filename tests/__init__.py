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

from numpy import testing as nptest

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
        with self.assertRaises(LookupError):
            cldrive.extract_args(src)

    def test_extract_args_no_kernel(self):
        src = """__kernel void A();"""
        with self.assertRaises(LookupError):
            cldrive.extract_args(src)

    def test_extract_args_no_args(self):
        src = """__kernel void A() {}"""
        args = cldrive.extract_args(src)
        self.assertEqual(len(args), 0)

#     def test_make_data_no_args(self):
#         src = '''\
# __kernel void A() {}
# '''
#         data = cldrive.make_data(
#             src, size=32, data_generator=cldrive.Generator.ZEROS)

#         self.assertEqual(data.shape, (0,))

#     def test_make_data_1_arg(self):
#         src = '''\
# __kernel void A(__global int* data) {}
# '''
#         data = cldrive.make_data(
#             src, size=1, data_generator=cldrive.Generator.ZEROS)

#         self.assertEqual(data.shape, (1, 1))
#         self.assertEqual(data[0].dtype, np.int32)
#         self.assertEqual(data[0][0], 0)

#     def test_make_data_3_args(self):
#         src = '''\
# __kernel void A(__global float* a, const int b, __local int* c) {}
# '''
#         data = cldrive.make_data(
#             src, size=32, data_generator=cldrive.Generator.ZEROS)

#         self.assertEqual(data.shape, (2,))
#         self.assertEqual(data[0].shape, (32,))
#         self.assertEqual(data[1].shape, (1,))
#         self.assertEqual(data[0].dtype, np.float32)
#         self.assertEqual(data[1].dtype, np.int32)

#     def test_make_data_vector_types(self):
#         src = '''\
# kernel void A(global float4* a, global int2* b, const int3 c) {}
# '''
#         data = cldrive.make_data(
#             src, 512, data_generator=cldrive.Generator.ZEROS)

#         self.assertEqual(data.shape, (3,))
#         self.assertEqual(data[0].shape, (512 * 4,))
#         self.assertEqual(data[1].shape, (512 * 2,))
#         self.assertEqual(data[2].shape, (3,))
#         self.assertEqual(data[0].dtype, np.float32)
#         self.assertEqual(data[1].dtype, np.int32)
#         self.assertEqual(data[2].dtype, np.int32)

    def test_make_env_not_found(self):
        with self.assertRaises(LookupError):
            cldrive.make_env(platform_id=9999999, device_id=9999999)

#     def test_run_1(self):
#         src = '''\
# __kernel void A(__global int* data) {
#     int tid = get_global_id(0);
#     data[tid] *= 2;
# }
# '''
#         env = cldrive.make_env()

#         outputs = cldrive.run_kernel(src, data_generator=cldrive.Generator.SEQ,
#                                      gsize=cldrive.NDRange(4,1,1),
#                                      lsize=cldrive.NDRange(1,1,1),
#                                      env=env)
#         nptest.assert_equal(outputs, [[0,2,4,6]])

#         outputs = cldrive.run_kernel(src, data_generator=cldrive.Generator.ZEROS,
#                                      gsize=cldrive.NDRange(8,2,1),
#                                      lsize=cldrive.NDRange(1,1,1),
#                                      env=env)
#         nptest.assert_equal(outputs, [[0] * 16])

#     def test_run_2(self):
#         src = '''\
# __kernel void A(__global float* data, const float b) {
#     int xid = get_global_id(0);
#     int yid = get_global_id(1) * get_global_size(0);
#     data[yid + xid] = data[yid + xid] + b;
# }
# '''
#         env = cldrive.make_env()

#         outputs = cldrive.run_kernel(src, data_generator=cldrive.Generator.SEQ,
#                                      gsize=cldrive.NDRange(4,1,1),
#                                      lsize=cldrive.NDRange(2,1,1),
#                                      env=env)
#         nptest.assert_almost_equal(outputs[0], [4,5,6,7])
#         nptest.assert_almost_equal(outputs[1], [4])

#         # same again, but scale the buffer this time
#         outputs = cldrive.run_kernel(src, data_generator=cldrive.Generator.SEQ,
#                                      gsize=cldrive.NDRange(4,1,1),
#                                      lsize=cldrive.NDRange(2,1,1),
#                                      env=env, buf_scale=2.0)
#         nptest.assert_almost_equal(outputs[0], [4,5,6,7,4,5,6,7])
#         nptest.assert_almost_equal(outputs[1], [4])

#         outputs = cldrive.run_kernel(src, data_generator=cldrive.Generator.ZEROS,
#                                      gsize=cldrive.NDRange(8,2,1),
#                                      lsize=cldrive.NDRange(2,2,1),
#                                      env=env)
#         nptest.assert_almost_equal(outputs[0], [16] * 16)
#         nptest.assert_almost_equal(outputs[1], [16])

# TODO: Difftest against cl_launcher from CLSmith for a CLSmith kernel.
