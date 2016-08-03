from unittest import TestCase,skip,skipIf
import tests

import sys
import os

import labm8
from labm8 import fs

import smith
from smith import drive
from smith import config as cfg

source1 = """
__kernel void A(__global float* a, __global float* b, const int c) {
    int d = get_global_id(0);

    if (d < c) {
        a[d] += b[d];
    }
}
"""

source1_bad = """
__kernel void A(__global float* a) {
    UNDEFINED_SYMBOLS;
}
"""

@skipIf(not cfg.host_has_opencl(), "no OpenCL support in host")
class TestKernelDriver(TestCase):
    def test_foo(self):
        self.assertEqual(True, True)


if __name__ == '__main__':
    main()
