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

import cldrive


class TestArgs(TestCase):
    def test_extract_args_syntax_error(self):
        src = """ kernel float float A(asd/ ) {}"""
        with self.assertRaises(cldrive.OpenCLValueError):
            cldrive.extract_args(src)
        # can be caught more generally as a ValueError
        with self.assertRaises(ValueError):
            cldrive.extract_args(src)

    def test_extract_args_multiple_kernels(self):
        src = """
            __kernel void A(__global int* a) {}
            __kernel void B(const int e) {}
        """
        with self.assertRaises(LookupError):
            cldrive.extract_args(src)

    def test_extract_args_no_kernels(self):
        src = """__kernel void A();"""
        with self.assertRaises(LookupError):
            cldrive.extract_args(src)

    def test_extract_args_struct(self):
        src = """
            struct C;
            __kernel void A(struct C a) {}
        """
        with self.assertRaises(ValueError):
            cldrive.extract_args(src)

    def test_extract_args_local_global_qual(self):
        src = """
            __kernel void A(global local float* a) {}
        """
        with self.assertRaises(cldrive.OpenCLValueError):
            cldrive.extract_args(src)

    def test_extract_args_no_qual(self):
        src = """
            __kernel void A(float* a) {}
        """
        with self.assertRaises(cldrive.OpenCLValueError):
            cldrive.extract_args(src)

    def test_extract_args_no_args(self):
        src = """__kernel void A() {}"""
        args = cldrive.extract_args(src)
        self.assertEqual(len(args), 0)

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
        self.assertTrue(args[0].is_const)
        self.assertTrue(args[0].is_pointer)
        self.assertEqual(args[0].typename, "int")
        self.assertEqual(args[0].bare_type, "int")

    def test_extract_args_address_spaces(self):
        src = """
        kernel void A(global int* a, local int* b, constant int* c, const int d) {}
        """
        args = cldrive.extract_args(src)
        self.assertEqual(len(args), 4)
        self.assertEqual(args[0].address_space, "global")
        self.assertEqual(args[1].address_space, "local")
        self.assertEqual(args[2].address_space, "constant")
        self.assertEqual(args[3].address_space, "private")

    def test_extract_args_no_address_space(self):
        src = """
        kernel void A(int* a) {}
        """
        with self.assertRaises(cldrive.OpenCLValueError):
            args = cldrive.extract_args(src)

    def test_extract_args_multiple_address_spaces(self):
        src = """
        kernel void A(global local int* a) {}
        """
        with self.assertRaises(cldrive.OpenCLValueError):
            args = cldrive.extract_args(src)

    def test_extract_args_properties(self):
        src = """
        kernel void A(const global int* a, global const float* b,
                      local float4 *const c, const int d, float2 e) {}
        """
        args = cldrive.extract_args(src)
        self.assertEqual(len(args), 5)
        self.assertEqual(args[0].is_pointer, True)
        self.assertEqual(args[0].address_space, "global")
        self.assertEqual(args[0].typename, "int")
        self.assertEqual(args[0].name, "a")
        self.assertEqual(args[0].bare_type, "int")
        self.assertEqual(args[0].is_vector, False)
        self.assertEqual(args[0].vector_width, 1)
        self.assertEqual(args[0].is_const, True)

        self.assertEqual(args[1].is_pointer, True)
        self.assertEqual(args[1].address_space, "global")
        self.assertEqual(args[1].typename, "float")
        self.assertEqual(args[1].name, "b")
        self.assertEqual(args[1].bare_type, "float")
        self.assertEqual(args[1].is_vector, False)
        self.assertEqual(args[1].vector_width, 1)
        self.assertEqual(args[1].is_const, True)

        self.assertEqual(args[2].is_pointer, True)
        self.assertEqual(args[2].address_space, "local")
        self.assertEqual(args[2].typename, "float4")
        self.assertEqual(args[2].name, "c")
        self.assertEqual(args[2].bare_type, "float")
        self.assertEqual(args[2].is_vector, True)
        self.assertEqual(args[2].vector_width, 4)
        self.assertEqual(args[2].is_const, False)

    def test_extract_args_struct(self):
        src = """
        struct s { int a; };

        kernel void A(global struct s *a) {}
        """
        # we can't handle structs yet
        with self.assertRaises(cldrive.OpenCLValueError):
            cldrive.extract_args(src)


if __name__ == "__main__":
    main()
