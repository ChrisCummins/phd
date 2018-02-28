import pytest

from absl import app

import numpy as np
from numpy import testing as nptest

from gpu import cldrive
from gpu.cldrive.tests.lib import *


@pytest.mark.skip(reason="FIXME(cec)")
def test_zeros():
    src = "kernel void A(global float* a) {}"

    outputs = cldrive.zeros(src, 64)
    outputs_gs = [np.zeros(64)]

    almost_equal(outputs, outputs_gs)


@pytest.mark.skip(reason="FIXME(cec)")
def test_ones():
    src = "kernel void A(global float* a, const int b) {}"

    outputs = cldrive.ones(src, 1024)
    outputs_gs = [np.ones(1024), [1024]]

    almost_equal(outputs, outputs_gs)


@pytest.mark.skip(reason="FIXME(cec)")
def test_arange():
    src = "kernel void A(global float* a, local float* b, const int c) {}"

    outputs = cldrive.arange(src, 512, scalar_val=0)
    outputs_gs = [np.arange(512), [0]]

    almost_equal(outputs, outputs_gs)


def test_rand():
    src = "kernel void A(global float* a, global float* b) {}"

    outputs = cldrive.rand(src, 16)

    # we can't test the actual values
    assert outputs.shape == (2, 16)


@pytest.mark.skip(reason="FIXME(cec)")
def test_data_unchanged():
    src = "kernel void A(global int* a, global int* b, const int c) {}"

    inputs = cldrive.rand(src, 16)
    outputs = cldrive.drive(ENV, src, inputs, gsize=(16,1,1), lsize=(1,1,1))

    almost_equal(outputs, inputs)


@pytest.mark.skip(reason="FIXME(cec)")
def test_data_zerod():
    # zero-ing a randomly initialized array
    src = "kernel void A(global int* a) { a[get_global_id(0)] = 0; }"

    inputs = cldrive.rand(src, 16)
    outputs = cldrive.drive(ENV, src, inputs, gsize=(16,1,1), lsize=(4,1,1))

    almost_equal(outputs, [np.zeros(16)])


@pytest.mark.skip(reason="FIXME(cec)")
def test_vector_input_switch():
    src = """
    kernel void A(global int2* a) {
        const int tid = get_global_id(0);

        const int tmp = a[tid].x;
        a[tid].x = a[tid].y;
        a[tid].y = tmp;
    }
    """

    inputs = cldrive.arange(src, 4)
    outputs_gs = [[1, 0, 3, 2, 5, 4, 7, 6]]

    outputs = cldrive.drive(ENV, src, inputs, gsize=(4,1,1), lsize=(4,1,1))

    almost_equal(outputs, outputs_gs)


def main(argv):  # pylint: disable=missing-docstring
    del argv
    sys.exit(pytest.main([__file__, "-v"]))


if __name__ == "__main__":
    app.run(main)
