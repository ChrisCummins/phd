#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
import numpy as np

from clgen import clutil

source1 = """
__kernel void A(__global float* a,    __global float* b, const int c) {
    int d = get_global_id(0);

    if (d < c) {
        a[d] += 1;
    }
}
"""
source1_prototype = "__kernel void A(__global float* a, __global float* b, const int c) {"

source2 = """
__kernel void AB(__global float* a, __global float* b, __local int* c) {
    int d = get_global_id(0);

    for (int i = 0; i < d * 1000; ++i)
        a[d] += 1;
}
"""
source2_prototype = ("__kernel void AB(__global float* a, "
                     "__global float* b, __local int* c) {")

source3 = """
__kernel void C(__global int* a, __global int* b,

                const int c, const int d) {
    int e = get_global_id(0);
    a[e] = b[e] + c * d;
}
"""
source3_prototype = """
__kernel void C(__global int* a, __global int* b, const int c, const int d) {
""".strip()

test_sources = [source1, source2, source3]
test_prototypes = [source1_prototype, source2_prototype, source3_prototype]
test_names = ["A", "AB", "C"]
test_args = [
    ['__global float* a', '__global float* b', 'const int c'],
    ['__global float* a', '__global float* b', '__local int* c'],
    ['__global int* a', '__global int* b', 'const int c', 'const int d']
]
test_arg_globals = [[True, True, False],
                    [True, True, False],
                    [True, True, False, False]]
test_arg_locals = [[False, False, False],
                   [False, False, True],
                   [False, False, False, False]]
test_arg_names = [['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c', 'd']]
test_arg_types = [['float*', 'float*', 'int'],
                  ['float*', 'float*', 'int*'],
                  ['int*', 'int*', 'int', 'int']]
test_inputs = list(zip(test_sources, test_prototypes))


# OpenCLUtil


def test_get_kernel_prototype():
    for source, prototype in test_inputs:
        assert prototype == str(clutil.extract_prototype(source))


def test_strip_attributes():
    assert "" == clutil.strip_attributes(
        "__attribute__((reqd_work_group_size(64,1,1)))")

    out = "foobar"
    tin = "foo__attribute__((reqd_work_group_size(WG_SIZE,1,1)))bar"
    assert out == clutil.strip_attributes(tin)

    out = "typedef  unsigned char uchar8;"
    tin = ("typedef __attribute__((ext_vector_type(8))) "
           "unsigned char uchar8;")
    assert out == clutil.strip_attributes(tin)

    out = ("typedef  unsigned char uchar8;\n"
           "typedef  unsigned char uchar8;")
    tin = ("typedef __attribute__  ((ext_vector_type(8))) "
           "unsigned char uchar8;\n"
           "typedef __attribute__((reqd_work_group_size(64,1,1))) "
           "unsigned char uchar8;")
    assert out == clutil.strip_attributes(tin)


# KernelPrototype


def test_from_source():
    for source, prototype in test_inputs:
        assert prototype == str(clutil.KernelPrototype.from_source(source))


def test_prototype_name():
    for source, name in zip(test_sources, test_names):
        p = clutil.KernelPrototype.from_source(source)
        assert name == p.name


def test_prototye_args():
    for source, args in zip(test_sources, test_args):
        p = clutil.KernelPrototype.from_source(source)
        assert args == [str(x) for x in p.args]


def test_args_names():
    for source, argnames in zip(test_sources, test_arg_names):
        p = clutil.KernelPrototype.from_source(source)
        assert argnames == [x.name for x in p.args]


def test_args_types():
    for source, type_ in zip(test_sources, test_arg_types):
        p = clutil.KernelPrototype.from_source(source)
        assert type_ == [x.type for x in p.args]


def test_args_is_global():
    for source, isglobal in zip(test_sources, test_arg_globals):
        p = clutil.KernelPrototype.from_source(source)
        assert isglobal == [x.is_global for x in p.args]


def test_args_is_local():
    for source, islocal in zip(test_sources, test_arg_locals):
        p = clutil.KernelPrototype.from_source(source)
        assert islocal == [x.is_local for x in p.args]


# KernelArg


def test_string():
    assert clutil.KernelArg("global float4* a").string == "global float4* a"
    assert clutil.KernelArg("const int b").string == "const int b"


def test_components():
    assert (clutil.KernelArg("__global float4* a").components ==
            ["__global", "float4*", "a"])
    assert clutil.KernelArg("const int b").components == ["const", "int", "b"]
    assert clutil.KernelArg("const __restrict int c").components == ["const", "int", "c"]


def test_kernelarg_name():
    assert clutil.KernelArg("__global float4* a").name == "a"
    assert clutil.KernelArg("const int b").name == "b"


def test_type():
    assert clutil.KernelArg("__global float4* a").type == "float4*"
    assert clutil.KernelArg("const int b").type == "int"


def test_is_restrict():
    assert not clutil.KernelArg("__global float4* a").is_restrict
    assert clutil.KernelArg("const restrict int b").is_restrict


def test_qualifiers():
    assert clutil.KernelArg("__global float4* a").qualifiers == ["__global"]
    assert clutil.KernelArg("const int b").qualifiers == ["const"]
    assert clutil.KernelArg("int c").qualifiers == []


def test_is_pointer():
    assert clutil.KernelArg("__global float4* a").is_pointer
    assert not clutil.KernelArg("const int b").is_pointer


def test_is_vector():
    assert clutil.KernelArg("__global float4* a").is_vector
    assert not clutil.KernelArg("const int b").is_vector


def test_vector_width():
    assert clutil.KernelArg("__global float4* a").vector_width == 4
    assert clutil.KernelArg("const int32 b").vector_width == 32
    assert clutil.KernelArg("const int c").vector_width == 1


def test_bare_type():
    assert clutil.KernelArg("__global float4* a").bare_type == "float"
    assert clutil.KernelArg("const int b").bare_type == "int"


def test_is_const():
    assert not clutil.KernelArg("__global float4* a").is_const
    assert clutil.KernelArg("const int b").is_const


def test_is_global():
    assert clutil.KernelArg("__global float4* a").is_global
    assert not clutil.KernelArg("const int b").is_global


def test_is_local():
    assert clutil.KernelArg("__local float4* a").is_local
    assert not clutil.KernelArg("const int b").is_local


def test_numpy_type():
    assert clutil.KernelArg("__local float4* a").numpy_type == np.float32
    assert clutil.KernelArg("const int b").numpy_type == np.int32


def test_arg1():
    a = clutil.KernelArg("__global float* a")
    assert "float*" == a.type
    assert "float" == a.bare_type
    assert a.is_pointer
    assert a.is_global
    assert not a.is_local
    assert not a.is_const
    assert np.float32 == a.numpy_type
    assert 1 == a.vector_width


def test_arg2():
    a = clutil.KernelArg("__global float4* a")
    assert "float4*" == a.type
    assert "float" == a.bare_type
    assert a.is_pointer
    assert a.is_global
    assert not a.is_local
    assert not a.is_const
    assert np.float32 == a.numpy_type
    assert 4 == a.vector_width


def test_arg3():
    a = clutil.KernelArg("const unsigned int z")
    assert "unsigned int" == a.type
    assert "unsigned int" == a.bare_type
    assert not a.is_pointer
    assert not a.is_global
    assert not a.is_local
    assert a.is_const
    assert np.uint32 == a.numpy_type
    assert 1 == a.vector_width


def test_arg4():
    a = clutil.KernelArg("const uchar16 z")
    assert "uchar16" == a.type
    assert "uchar" == a.bare_type
    assert not a.is_pointer
    assert not a.is_global
    assert not a.is_local
    assert a.is_const
    assert np.uint8 == a.numpy_type
    assert 16 == a.vector_width
