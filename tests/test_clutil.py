from unittest import TestCase,skip
import tests

import numpy as np
import sys
import os

import labm8
from labm8 import fs

import clgen
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
source2_prototype = "__kernel void AB(__global float* a, __global float* b, __local int* c) {"

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
test_arg_names = [['a', 'b', 'c'], ['a', 'b', 'c'], ['a', 'b', 'c', 'd']]
test_arg_types = [['float*', 'float*', 'int'],
                  ['float*', 'float*', 'int*'],
                  ['int*', 'int*', 'int', 'int']]
test_arg_globals = [[True, True, False],
                    [True, True, False],
                    [True, True, False, False]]
test_arg_locals = [[False, False, False],
                   [False, False, True],
                   [False, False, False, False]]
test_inputs = zip(test_sources, test_prototypes)


class TestOpenCLUtil(TestCase):
    def test_get_kernel_prototype(self):
        for source,prototype in test_inputs:
            self.assertEqual(prototype,
                             str(clutil.extract_prototype(source)))

    def test_strip_attributes(self):
        self.assertEqual("", clutil.strip_attributes(
            "__attribute__((reqd_work_group_size(64,1,1)))"))

        out = "foobar"
        tin = "foo__attribute__((reqd_work_group_size(WG_SIZE,1,1)))bar"
        self.assertEqual(out, clutil.strip_attributes(tin))

        out = "typedef  unsigned char uchar8;"
        tin = "typedef __attribute__((ext_vector_type(8))) unsigned char uchar8;"
        self.assertEqual(out, clutil.strip_attributes(tin))

        out = ("typedef  unsigned char uchar8;\n"
               "typedef  unsigned char uchar8;")
        tin = ("typedef __attribute__  ((ext_vector_type(8))) unsigned char uchar8;\n"
               "typedef __attribute__((reqd_work_group_size(64,1,1))) unsigned char uchar8;")
        self.assertEqual(out, clutil.strip_attributes(tin))


class TestKernelPrototype(TestCase):
    def test_from_source(self):
        for source, prototype in test_inputs:
            self.assertEqual(prototype,
                             str(clutil.KernelPrototype.from_source(source)))

    def test_kernel_name(self):
        for source,name in zip(test_sources, test_names):
            p = clutil.KernelPrototype.from_source(source)
            self.assertEqual(name, p.name)

    def test_kernel_args(self):
        for source,args in zip(test_sources, test_args):
            p = clutil.KernelPrototype.from_source(source)
            self.assertEqual(args, [str(x) for x in p.args])


class TestKernelArg(TestCase):
    def test_arg_names(self):
        for source,argnames in zip(test_sources, test_arg_names):
            p = clutil.KernelPrototype.from_source(source)
            self.assertEqual(argnames, [x.name for x in p.args])

    def test_arg_names(self):
        for source,type in zip(test_sources, test_arg_types):
            p = clutil.KernelPrototype.from_source(source)
            self.assertEqual(type, [x.type for x in p.args])

    def test_arg_is_global(self):
        for source,isglobal in zip(test_sources, test_arg_globals):
            p = clutil.KernelPrototype.from_source(source)
            self.assertEqual(isglobal, [x.is_global for x in p.args])

    def test_arg_is_local(self):
        for source,islocal in zip(test_sources, test_arg_locals):
            p = clutil.KernelPrototype.from_source(source)
            self.assertEqual(islocal, [x.is_local for x in p.args])

    def test_arg(self):
        a = clutil.KernelArg("__global float* a")
        self.assertEqual("float*", a.type)
        self.assertEqual("float", a.bare_type)
        self.assertTrue(a.is_pointer)
        self.assertTrue(a.is_global)
        self.assertFalse(a.is_local)
        self.assertFalse(a.is_const)
        self.assertIs(np.float32, a.numpy_type)
        self.assertEqual(1, a.vector_width)

        a = clutil.KernelArg("__global float4* a")
        self.assertEqual("float4*", a.type)
        self.assertEqual("float", a.bare_type)
        self.assertTrue(a.is_pointer)
        self.assertTrue(a.is_global)
        self.assertFalse(a.is_local)
        self.assertFalse(a.is_const)
        self.assertIs(np.float32, a.numpy_type)
        self.assertEqual(4, a.vector_width)

        a = clutil.KernelArg("const unsigned int z")
        self.assertEqual("unsigned int", a.type)
        self.assertEqual("unsigned int", a.bare_type)
        self.assertFalse(a.is_pointer)
        self.assertFalse(a.is_global)
        self.assertFalse(a.is_local)
        self.assertTrue(a.is_const)
        self.assertIs(np.uint32, a.numpy_type)
        self.assertEqual(1, a.vector_width)

        a = clutil.KernelArg("const uchar16 z")
        self.assertEqual("uchar16", a.type)
        self.assertEqual("uchar", a.bare_type)
        self.assertFalse(a.is_pointer)
        self.assertFalse(a.is_global)
        self.assertFalse(a.is_local)
        self.assertTrue(a.is_const)
        self.assertIs(np.uint8, a.numpy_type)
        self.assertEqual(16, a.vector_width)
