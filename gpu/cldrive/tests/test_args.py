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

import cldrive

from lib import *


def test_preprocess():
    pp = cldrive.preprocess("kernel void A() {}")
    assert pp.split("\n")[-2] == "kernel void A() {}"


def test_parse():
    ast = cldrive.parse("kernel void A() {}")
    assert len(ast.children()) >= 1


def test_extract_args_syntax_error():
    src = "kernel void A(@!"
    with pytest.raises(cldrive.OpenCLValueError):
        cldrive.parse(src)

    # OpenCLValueError extends ValueError
    with pytest.raises(ValueError):
        cldrive.parse(src)


def test_parse_preprocess():
    src = """
    #define DTYPE float
    kernel void A(global DTYPE *a) {}
    """

    pp = cldrive.preprocess(src)
    ast = cldrive.parse(pp)
    assert len(ast.children()) >= 1


def test_parse_header():
    src = """
    #include "header.h"
    kernel void A(global DTYPE* a) {
      a[get_global_id(0)] = DOUBLE(a[get_global_id(0)]);
    }
    """
    pp = cldrive.preprocess(src, include_dirs=[data_path("")])
    ast = cldrive.parse(pp)
    assert len(ast.children()) >= 1


def test_extract_args():
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

    assert len(args) == 5
    assert args[0].is_const
    assert args[0].is_pointer
    assert args[0].typename == "int"
    assert args[0].bare_type == "int"


def test_extract_args_no_declaration():
    with pytest.raises(LookupError):
        cldrive.extract_args("")


def test_extract_args_no_definition():
    src = "kernel void A();"
    with pytest.raises(LookupError):
        cldrive.extract_args(src)


def test_extract_args_multiple_kernels():
    src = "kernel void A() {} kernel void B() {}"
    with pytest.raises(LookupError):
        cldrive.extract_args(src)


def test_extract_args_struct():
    src = "struct C; kernel void A(struct C a) {}"
    with pytest.raises(ValueError):
        cldrive.extract_args(src)


def test_extract_args_local_global_qualified():
    src = "kernel void A(global local int* a) {}"
    with pytest.raises(cldrive.OpenCLValueError):
        cldrive.extract_args(src)


def test_extract_args_no_qualifiers():
    src = "kernel void A(float* a) {}"
    with pytest.raises(cldrive.OpenCLValueError):
        cldrive.extract_args(src)


def test_extract_args_no_args():
    src = "kernel void A() {}"
    assert len(cldrive.extract_args(src)) == 0


def test_extract_args_address_spaces():
    src = """
    kernel void A(global int* a, local int* b, constant int* c, const int d) {}
    """
    args = cldrive.extract_args(src)
    assert len(args) == 4
    assert args[0].address_space == "global"
    assert args[1].address_space == "local"
    assert args[2].address_space == "constant"
    assert args[3].address_space == "private"


def test_extract_args_no_address_space():
    src = """
    kernel void A(int* a) {}
    """
    with pytest.raises(cldrive.OpenCLValueError):
        args = cldrive.extract_args(src)


def test_extract_args_multiple_address_spaces():
    src = """
    kernel void A(global local int* a) {}
    """
    with pytest.raises(cldrive.OpenCLValueError):
        args = cldrive.extract_args(src)


def test_extract_args_properties():
    src = """
    kernel void A(const global int* a, global const float* b,
                  local float4 *const c, const int d, float2 e) {}
    """
    args = cldrive.extract_args(src)
    assert len(args) == 5
    assert args[0].is_pointer == True
    assert args[0].address_space == "global"
    assert args[0].typename == "int"
    assert args[0].name == "a"
    assert args[0].bare_type == "int"
    assert args[0].is_vector == False
    assert args[0].vector_width == 1
    assert args[0].is_const == True

    assert args[1].is_pointer ==  True
    assert args[1].address_space == "global"
    assert args[1].typename == "float"
    assert args[1].name == "b"
    assert args[1].bare_type == "float"
    assert args[1].is_vector ==  False
    assert args[1].vector_width ==  1
    assert args[1].is_const ==  True

    assert args[2].is_pointer == True
    assert args[2].address_space == "local"
    assert args[2].typename == "float4"
    assert args[2].name == "c"
    assert args[2].bare_type == "float"
    assert args[2].is_vector == True
    assert args[2].vector_width == 4
    assert args[2].is_const == False


def test_extract_args_struct():
    src = """
    struct s { int a; };

    kernel void A(global struct s *a) {}
    """
    # we can't handle structs yet
    with pytest.raises(cldrive.OpenCLValueError):
        cldrive.extract_args(src)

def test_extract_args_preprocess():
    src = """
    #define DTYPE float
    kernel void A(global DTYPE *a) {}
    """

    pp = cldrive.preprocess(src)
    args = cldrive.extract_args(pp)
    assert args[0].typename == 'float'
