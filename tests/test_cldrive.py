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
from unittest import TestCase, skip, skipIf
import tests

from copy import deepcopy
from functools import partial

import numpy as np
import os
import sys

import labm8
from labm8 import fs

import clgen
from clgen import cldrive
from clgen import clutil
from clgen import config as cfg

if cfg.USE_OPENCL:
    import pyopencl as cl

source1 = """
__kernel void A(__global float* a, __global float* b, const int c) {
    int d = get_global_id(0);

    if (d < c) {
        a[d] = b[d] * 2;
    }
}
"""

source2 = """
__kernel void A(__global float* a, __global int* b, const int c) {
    int d = get_global_id(0);
    int e = 0;

    for (int f = 0; f < c; ++f)
      e += b[d];

    a[d] = (float)e;
}
"""

source3 = """
__kernel void A(__global float* a, __local float* b, const int c) {
    int d = get_global_id(0);

    if (d < c)
      b[d] = a[d];

    barrier(1);

    a[d] = a[d] + b[d];
}
"""

source1_E_BAD_CODE = """
__kernel void A(__global float* a) {
    UNDEFINED_SYMBOLS;
}
"""

source2_E_BAD_CODE = """
__kernel void A(__global float* a) {
"""

source1_E_NON_TERMINATING = """
__kernel void A(__global float* a) {
  int b = get_global_id(0);

  while (1)
    a[b] += 1;
}
"""

#
source1_E_NONDETERMINISTIC = """
__kernel void A(__global float* a) {
  int b = get_global_id(0);

  a[b] = a[b-1] + a[b+1];
}
"""

# This kernel always writes the same value to argument 'a'.
source1_E_INPUT_INSENSITIVE = """
__kernel void A(__global int* a) {
  int b = get_global_id(0);
  a[b] = 0;
}
"""

source1_E_NO_OUTPUTS = """
__kernel void A(__global int* a) {}
"""

@skipIf(not cfg.USE_OPENCL, "no OpenCL")
class TestKernelDriver(TestCase):
    def setUp(self):
        self._devtype = cl.device_type.GPU
        self._ctx, self._queue = cldrive.init_opencl(devtype=self._devtype)

    def test_build_program(self):
        prog = cldrive.KernelDriver.build_program(self._ctx, source1)
        self.assertIsInstance(prog, cl.Program)

        driver = cldrive.KernelDriver(self._ctx, source1)
        self.assertIsInstance(driver, cldrive.KernelDriver)

        prog = cldrive.KernelDriver.build_program(self._ctx, source2)
        self.assertIsInstance(prog, cl.Program)

        prog = cldrive.KernelDriver.build_program(self._ctx, source3)
        self.assertIsInstance(prog, cl.Program)

        driver = cldrive.KernelDriver(self._ctx, source1)
        self.assertIsInstance(driver, cldrive.KernelDriver)

        with self.assertRaises(cldrive.E_BAD_CODE):
            cldrive.KernelDriver.build_program(self._ctx, source1_E_BAD_CODE)

        with self.assertRaises(cldrive.E_BAD_CODE):
            cldrive.KernelDriver.build_program(self._ctx, source2_E_BAD_CODE)

    def test_source(self):
        driver = cldrive.KernelDriver(self._ctx, source1)
        self.assertEqual(source1, driver.source)

    def test_context(self):
        driver = cldrive.KernelDriver(self._ctx, source1)
        self.assertEqual(self._ctx, driver.context)

    def test_prototype(self):
        driver = cldrive.KernelDriver(self._ctx, source1)
        self.assertIsInstance(driver.prototype, clutil.KernelPrototype)

        self.assertEqual(3, len(driver.prototype.args))

        self.assertEqual("a", driver.prototype.args[0].name)
        self.assertEqual("b", driver.prototype.args[1].name)
        self.assertEqual("c", driver.prototype.args[2].name)

        self.assertEqual("float*", driver.prototype.args[0].type)
        self.assertEqual("float*", driver.prototype.args[1].type)
        self.assertEqual("int", driver.prototype.args[2].type)

        self.assertTrue(driver.prototype.args[0].is_global)
        self.assertTrue(driver.prototype.args[1].is_global)
        self.assertFalse(driver.prototype.args[2].is_global)

        self.assertTrue(driver.prototype.args[0].is_pointer)
        self.assertTrue(driver.prototype.args[1].is_pointer)
        self.assertFalse(driver.prototype.args[2].is_pointer)

    def test_name(self):
        driver = cldrive.KernelDriver(self._ctx, source1)
        self.assertEqual("A", driver.name)

    def test_kernel(self):
        # Run kernel:
        driver = cldrive.KernelDriver(self._ctx, source1)
        kernel = driver.kernel
        sz = 8
        host = np.random.rand(sz).astype(np.float32)
        dev = [
            cl.Buffer(self._ctx, cl.mem_flags.WRITE_ONLY, host.nbytes),
            cl.Buffer(self._ctx,
                      cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                      hostbuf=host),
            np.int32(sz)
        ]
        kernel(self._queue, (sz,), None, *dev)
        result = np.zeros(sz).astype(np.float32)
        cl.enqueue_copy(self._queue, result, dev[0], is_blocking=True)
        for x, y in zip(host, result):
            self.assertAlmostEqual(x * 2, y)

    def test_call(self):
        driver = cldrive.KernelDriver(self._ctx, source1)
        k = partial(driver, self._queue)

        A = cldrive.KernelPayload.create_sequential(driver, 8)
        B = k(A)

        self.assertEqual(1, len(driver.runtimes))
        self.assertEqual(1, len(driver.wgsizes))
        self.assertTrue(driver.wgsizes[0] > 0)
        self.assertEqual(1, len(driver.transfers))
        self.assertEqual(A.transfersize, driver.transfers[0])
        self.assertTrue(A != B)

        C = k(A)

        self.assertEqual(2, len(driver.runtimes))
        self.assertEqual(2, len(driver.wgsizes))
        self.assertEqual(2, len(driver.transfers))
        self.assertEqual(A.transfersize, driver.transfers[-1])
        self.assertTrue(C != A)
        self.assertTrue(C == B)

        D = k(A)

        self.assertEqual(3, len(driver.runtimes))
        self.assertEqual(3, len(driver.wgsizes))
        self.assertEqual(3, len(driver.transfers))
        self.assertEqual(A.transfersize, driver.transfers[-1])
        self.assertTrue(D != A)
        self.assertTrue(D == C)

        # workgroup size should be consistent across all three runs
        self.assertTrue(all(x == driver.wgsizes[0]
                        for x in driver.wgsizes[1:]))

    @skip("how long you got?")
    def test_non_terminating(self):
        driver = cldrive.KernelDriver(self._ctx, source1_E_NON_TERMINATING)
        k = partial(driver, self._queue)

        A = cldrive.KernelPayload.create_sequential(driver, 16)
        with self.assertRaises(cldrive.OpenCLDriverException):
            k(A)

    def test_out_of_resources(self):
        driver = cldrive.KernelDriver(self._ctx, source1_E_NON_TERMINATING)
        wayyyyyy_too_big = 2**63

        with self.assertRaises(cldrive.E_BAD_ARGS):
            cldrive.KernelPayload.create_sequential(driver, wayyyyyy_too_big)

    def test_validate(self):
        driver = cldrive.KernelDriver(self._ctx, source1)
        driver.validate(self._queue, size=8)

    @skip("device-dependent")
    def test_validate_nondeterministic(self):
        driver = cldrive.KernelDriver(self._ctx, source1_E_NONDETERMINISTIC)
        with self.assertRaises(cldrive.E_NONDETERMINISTIC):
            for i in range(1000):
                driver.validate(self._queue, size=1024)

    @skip("FIXME: doesn't pass")
    def test_validate_input_insensitive(self):
        driver = cldrive.KernelDriver(self._ctx, source1_E_INPUT_INSENSITIVE)
        with self.assertRaises(cldrive.E_INPUT_INSENSITIVE):
            driver.validate(self._queue, size=8)

    def test_validate_no_outputs(self):
        driver = cldrive.KernelDriver(self._ctx, source1_E_NO_OUTPUTS)
        with self.assertRaises(cldrive.E_NO_OUTPUTS):
            driver.validate(self._queue, size=8)

    def test_profile(self):
        driver = cldrive.KernelDriver(self._ctx, source1)
        driver.profile(self._queue)
        self.assertEqual(10, len(driver.runtimes))

        driver = cldrive.KernelDriver(self._ctx, source2)
        driver.profile(self._queue)
        self.assertEqual(10, len(driver.runtimes))

        driver = cldrive.KernelDriver(self._ctx, source3)
        driver.profile(self._queue)
        self.assertEqual(10, len(driver.runtimes))


@skipIf(not cfg.USE_OPENCL, "no OpenCL")
class TestKernelPayload(TestCase):
    def setUp(self):
        self._devtype = cl.device_type.GPU
        self._ctx, self._queue = cldrive.init_opencl(devtype=self._devtype)
        self._driver1 = cldrive.KernelDriver(self._ctx, source1)
        self._driver2 = cldrive.KernelDriver(self._ctx, source2)
        self._driver3 = cldrive.KernelDriver(self._ctx, source3)

    def test_create_sequential(self):
        p = cldrive.KernelPayload.create_sequential(self._driver1, 8)
        self.assertIsInstance(p, cldrive.KernelPayload)
        self.assertEqual(3, len(p.kargs))
        self.assertIs(np.float32, p.args[0].numpy_type)
        self.assertIs(np.float32, p.args[1].numpy_type)
        self.assertIs(np.int32, p.args[2].numpy_type)

        self.assertIs(cl.Buffer, type(p.args[0].devdata))
        self.assertEqual(8, p.args[0].hostdata.size)
        self.assertIs(cl.Buffer, type(p.args[1].devdata))
        self.assertEqual(8, p.args[1].hostdata.size)
        self.assertIs(np.int32, type(p.args[2].devdata))
        self.assertEqual(8, p.args[2].devdata)

    def test_create_random(self):
        p = cldrive.KernelPayload.create_random(self._driver1, 8)
        self.assertIsInstance(p, cldrive.KernelPayload)
        self.assertEqual(3, len(p.kargs))
        self.assertIs(np.float32, p.args[0].numpy_type)
        self.assertIs(np.float32, p.args[1].numpy_type)
        self.assertIs(np.int32, p.args[2].numpy_type)

        self.assertIs(cl.Buffer, type(p.args[0].devdata))
        self.assertEqual(8, p.args[0].hostdata.size)
        self.assertIs(cl.Buffer, type(p.args[1].devdata))
        self.assertEqual(8, p.args[1].hostdata.size)
        self.assertIs(np.int32, type(p.args[2].devdata))
        self.assertEqual(8, p.args[2].devdata)

    def test_comparisons(self):
        p1 = cldrive.KernelPayload.create_sequential(self._driver1, 8)
        p2 = cldrive.KernelPayload.create_sequential(self._driver1, 8)
        self.assertEqual(p1, p2)

        p3 = cldrive.KernelPayload.create_sequential(self._driver1, 16)
        p4 = cldrive.KernelPayload.create_sequential(self._driver1, 16)
        self.assertNotEqual(p1, p3)
        self.assertEqual(p3, p4)

        p5 = cldrive.KernelPayload.create_random(self._driver1, 8)
        p6 = cldrive.KernelPayload.create_random(self._driver1, 8)
        self.assertNotEqual(p1, p5)
        self.assertNotEqual(p5, p6)


if __name__ == "__main__":
    main()
