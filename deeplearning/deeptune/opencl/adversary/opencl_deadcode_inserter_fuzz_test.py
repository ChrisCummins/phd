# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepTune is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepTune is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepTune.  If not, see <https://www.gnu.org/licenses/>.
"""Fuzz test for :opencl_deadcode_inserter."""
import pathlib
import random
import tempfile

import numpy as np
import pytest

from deeplearning.deepsmith.harnesses import cldrive
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deeptune.opencl.adversary import \
  opencl_deadcode_inserter as dci
from gpu.oclgrind import oclgrind
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

# The number of tests to run.
# TODO(cec): Can this be a flag?
FUZZ_TEST_COUNT = 10

KERNELS = np.array([
    """\
kernel void A(const int a) {
  int b = a;
}""", """\
kernel void A(global int * a) {
}""", """\
kernel void C(global int * a, const float b) {
}""", """\
kernel void AA() {
}""", """\
kernel void A(const int a, global int * b) {
  if (get_global_id(0) < a) {
    b[get_global_id(0)] += 1;
  }
}""", """\
kernel void A(int a, const __global float4* b, __global float4* c, __global float4* d, __global float4* e) {
  unsigned int f = get_local_id(0);
  unsigned int g = get_group_id(0);

  float4 h = b[g];

  float4 i = (1.0f - h) * 5.0f + h * 30.f;
  float4 j = (1.0f - h) * 1.0f + h * 100.f;
  float4 k = (1.0f - h) * 0.25f + h * 10.f;
  float4 l = k * (1.0f / (float)a);
  float4 m = 0.30f * sqrt(l);
  float4 n = 0.02f * l;
  float4 o = exp(n);
  float4 p = 1.0f / o;
  float4 q = exp(m);
  float4 r = 1.0f / q;
  float4 s = (o - r) / (q - r);
  float4 t = 1.0f - s;
  float4 u = s * p;
  float4 v = t * p;

  float4 w = i * exp(m * (2.0f * f - (float)a)) - j;
  d[f].x = w.x > 0 ? w.x : 0.0f;
  d[f].y = w.y > 0 ? w.y : 0.0f;
  d[f].z = w.z > 0 ? w.z : 0.0f;
  d[f].w = w.w > 0 ? w.w : 0.0f;

  barrier(1);

  for (int x = a; x > 0; x -= 2) {
    if (f < x) {
      e[f] = u * d[f] + v * d[f + 1];
    }
    barrier(1);

    if (f < x - 1) {
      d[f] = u * e[f] + v * e[f + 1];
    }
    barrier(1);
  }

  if (f == 0)
    c[g] = d[0];
}""", """\
__kernel void A(__global float* a, __global float* b, __global float* c, int d, int e, int f) {
  int g = get_global_id(0);
  int h = get_global_id(1);

  if ((h < d) && (g < e)) {
    int i;
    for (i = 0; i < f; i++) {
      c[h * e + g] += a[h * f + i] * b[i * e + g];
    }
  }
}"""
])


@pytest.mark.parametrize('i', range(FUZZ_TEST_COUNT))
def test_GenerateDeadcodeMutations_fuzz_test_batch(i: int):
  """Fuzz test the mutation generator.

  For each round of testing:
    Generate a batch of mutated kernels.
    For each mutated kernel:
      Create a DeepSmith testcase to compile and run the kernel.
      Drive the testcase using oclgrind.
      Ensure that the testcase completes.
  """
  del i  # unused

  # Select random parameters for test.
  kernels = [random.choice(KERNELS)
            ] + [k for k in KERNELS if random.random() < .5]
  seed = random.randint(0, 1e9)
  num_permutations_of_kernel = random.randint(1, 5)
  num_mutations_per_kernel_min = random.randint(1, 5)
  num_mutations_per_kernel_max = (num_mutations_per_kernel_min +
                                  random.randint(1, 5))
  num_mutations_per_kernel = (num_mutations_per_kernel_min,
                              num_mutations_per_kernel_max)

  app.Log(
      1, 'num_kernels=%d, seed=%d, num_permutations_of_kernel=%d, '
      'num_mutations_per_kernel=%s', len(kernels), seed,
      num_permutations_of_kernel, num_mutations_per_kernel)

  # Generate a batch of mutations.
  generator = dci.GenerateDeadcodeMutations(
      kernels=kernels,
      rand=np.random.RandomState(seed),
      num_permutations_of_kernel=num_permutations_of_kernel,
      num_mutations_per_kernel=num_mutations_per_kernel)

  for i, mutated_kernel in enumerate(generator):
    app.Log(1, "Testing mutated kernel: %s", mutated_kernel)

    # Create a DeepSmith testcase for the mutated kernel.
    testcase = deepsmith_pb2.Testcase(inputs={
        'lsize': '1,1,1',
        'gsize': '1,1,1',
        'src': mutated_kernel,
    })

    # Make a driver for the testcase.
    driver = cldrive.MakeDriver(testcase, optimizations=True)

    with tempfile.TemporaryDirectory(prefix='phd_') as d:
      # Compile the driver.
      binary = cldrive.CompileDriver(driver,
                                     pathlib.Path(d) / 'exe',
                                     0,
                                     0,
                                     timeout_seconds=60)
      # Execute the driver.
      proc = oclgrind.Exec([str(binary)])

    app.Log(1, "Testcase driver output: '%s'", proc.stderr.rstrip())
    assert not proc.returncode
    assert '[cldrive] Platform:' in proc.stderr
    assert '[cldrive] Device:' in proc.stderr
    assert '[cldrive] OpenCL optimizations: on\n' in proc.stderr
    assert '[cldrive] Kernel: "' in proc.stderr
    assert 'done.\n' in proc.stderr

  # Sanity check that the correct number of kernels have been generated.
  app.Log(1, 'Generated %d mutations', i + 1)
  assert i + 1 == len(kernels) * num_permutations_of_kernel


if __name__ == '__main__':
  test.Main()
