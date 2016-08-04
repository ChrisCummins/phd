from unittest import TestCase,skip,skipIf
import tests

import numpy as np
import os
import pyopencl as cl
import sys

import labm8
from labm8 import fs

import smith
from smith import clutil
from smith import config as cfg
from smith import drive

source1 = """
__kernel void A(__global float* a, __global float* b, const int c) {
    int d = get_global_id(0);

    if (d < c) {
        a[d] = b[d];
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

@skipIf(not cfg.host_has_opencl(), "no OpenCL support in host")
class TestKernelDriver(TestCase):
    def setUp(self):
        self._devtype = cl.device_type.GPU
        self._ctx, self._queue = drive.init_opencl(devtype=self._devtype)

    def test_build_program(self):
        prog = drive.KernelDriver.build_program(self._ctx, source1)
        self.assertIsInstance(prog, cl.Program)

        driver = drive.KernelDriver(self._ctx, source1)
        self.assertIsInstance(driver, drive.KernelDriver)

        prog = drive.KernelDriver.build_program(self._ctx, source2)
        self.assertIsInstance(prog, cl.Program)

        prog = drive.KernelDriver.build_program(self._ctx, source3)
        self.assertIsInstance(prog, cl.Program)

        driver = drive.KernelDriver(self._ctx, source1)
        self.assertIsInstance(driver, drive.KernelDriver)

        with self.assertRaises(drive.E_BAD_CODE):
            drive.KernelDriver.build_program(self._ctx, source1_E_BAD_CODE)

        with self.assertRaises(drive.E_BAD_CODE):
            drive.KernelDriver.build_program(self._ctx, source2_E_BAD_CODE)

    def test_source(self):
        driver = drive.KernelDriver(self._ctx, source1)
        self.assertEqual(source1, driver.source)

    def test_context(self):
        driver = drive.KernelDriver(self._ctx, source1)
        self.assertEqual(self._ctx, driver.context)

    def test_prototype(self):
        driver = drive.KernelDriver(self._ctx, source1)
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
        driver = drive.KernelDriver(self._ctx, source1)
        self.assertEqual("A", driver.name)

    def test_kernel(self):
        # Run kernel:
        driver = drive.KernelDriver(self._ctx, source1)
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
        for x,y in zip(host, result):
            self.assertAlmostEqual(x, y)


@skipIf(not cfg.host_has_opencl(), "no OpenCL support in host")
class TestKernelPayload(TestCase):
    def setUp(self):
        self._devtype = cl.device_type.GPU
        self._ctx, self._queue = drive.init_opencl(devtype=self._devtype)
        self._driver1 = drive.KernelDriver(self._ctx, source1)
        self._driver2 = drive.KernelDriver(self._ctx, source2)
        self._driver3 = drive.KernelDriver(self._ctx, source3)

    # def test_create_sequential(self):
    #     p = drive.KernelPayload.create_sequential(self._driver1, 8)
    #     self.assertIsInstance(p, drive.KernelPayload)
    #     self.assertEqual(3, len(p.kargs))

    # def test_comparisons(self):
    #     p1 = drive.KernelPayload.create_sequential(self._driver1, 8)
    #     p2 = drive.KernelPayload.create_sequential(self._driver1, 8)
    #     self.assertEqual(p1, p2)

    #     p3 = drive.KernelPayload.create_random(self._driver1, 8)
    #     self.assertNotEqual(p1, p3)


@skipIf(not cfg.host_has_opencl(), "no OpenCL support in host")
class TestDrive(TestCase):
    def setUp(self):
        self._devtype = cl.device_type.GPU

    # def test_kernel(self):
    #     drive.kernel(source1, size=8, devtype=self._devtype)


if __name__ == '__main__':
    main()
