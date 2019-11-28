# Copyright (c) 2017, 2018, 2019 Chris Cummins.
#
# DeepSmith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSmith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepSmith.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //deeplearning/deepsmith/harnesses/cl_launcher.py."""
import pytest

from deeplearning.deepsmith.harnesses import cl_launcher
from deeplearning.deepsmith.proto import deepsmith_pb2
from deeplearning.deepsmith.proto import harness_pb2
from deeplearning.deepsmith.proto import service_pb2
from gpu.cldrive.legacy import env
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

# A bare-bones CLSmith program.
CLSMITH_EXAMPLE_SRC = """
// -g 1,1,1 -l 1,1,1
#define int64_t long
#define uint64_t ulong
#define int_least64_t long
#define uint_least64_t ulong
#define int_fast64_t long
#define uint_fast64_t ulong
#define intmax_t long
#define uintmax_t ulong
#define int32_t int
#define uint32_t uint
#define int16_t short
#define uint16_t ushort
#define int8_t char
#define uint8_t uchar

#define INT64_MIN LONG_MIN
#define INT64_MAX LONG_MAX
#define INT32_MIN INT_MIN
#define INT32_MAX INT_MAX
#define INT16_MIN SHRT_MIN
#define INT16_MAX SHRT_MAX
#define INT8_MIN CHAR_MIN
#define INT8_MAX CHAR_MAX
#define UINT64_MIN ULONG_MIN
#define UINT64_MAX ULONG_MAX
#define UINT32_MIN UINT_MIN
#define UINT32_MAX UINT_MAX
#define UINT16_MIN USHRT_MIN
#define UINT16_MAX USHRT_MAX
#define UINT8_MIN UCHAR_MIN
#define UINT8_MAX UCHAR_MAX

#define transparent_crc(X, Y, Z) transparent_crc_(&crc64_context, X, Y, Z)

#define VECTOR(X , Y) VECTOR_(X, Y)
#define VECTOR_(X, Y) X##Y

#ifndef NO_GROUP_DIVERGENCE
#define GROUP_DIVERGE(x, y) get_group_id(x)
#else
#define GROUP_DIVERGE(x, y) (y)
#endif

#ifndef NO_FAKE_DIVERGENCE
#define FAKE_DIVERGE(x, y, z) (x - y)
#else
#define FAKE_DIVERGE(x, y, z) (z)
#endif

#include "CLSmith.h"

__kernel void entry(__global ulong *result) {
    uint64_t crc64_context = 0xFFFFFFFFFFFFFFFFUL;
    result[get_linear_global_id()] = crc64_context ^ 0xFFFFFFFFFFFFFFFFUL;
}
"""

# Test fixtures.


@pytest.fixture(scope='function')
def abc_testcase() -> deepsmith_pb2.Testcase():
  """A test fixture which returns a very simple test case."""
  return deepsmith_pb2.Testcase(
      toolchain='opencl',
      harness=deepsmith_pb2.Harness(name='cl_launcher'),
      inputs={
          'src': CLSMITH_EXAMPLE_SRC,
          'gsize': '1,1,1',
          'lsize': '1,1,1',
      })


@pytest.fixture(scope='function')
def abc_harness_config() -> harness_pb2.ClLauncherHarness:
  """A test fixture which returns an oclgrind harness config."""
  config = harness_pb2.ClLauncherHarness()
  config.opencl_env.extend([env.OclgrindOpenCLEnvironment().name])
  config.opencl_opt.extend([True])
  return config


@pytest.fixture(scope='function')
def abc_harness(abc_harness_config) -> cl_launcher.ClLauncherHarness:
  """A test fixture which returns an oclgrind harness."""
  return cl_launcher.ClLauncherHarness(abc_harness_config)


@pytest.fixture(scope='function')
def abc_run_testcases_request(abc_testcase,
                              abc_harness) -> harness_pb2.RunTestcasesRequest:
  """A test fixture which returns a RunTestcasesRequest for the abc_testcase."""
  return harness_pb2.RunTestcasesRequest(testbed=abc_harness.testbeds[0],
                                         testcases=[abc_testcase])


# Unit tests.

# ClLauncherHarness() tests.


def test_ClLauncherHarness_oclgrind_testbed():
  """Test that harness can be made from project-local oclgrind."""
  config = harness_pb2.ClLauncherHarness()
  config.opencl_env.extend([
      env.OclgrindOpenCLEnvironment().name,
      env.OclgrindOpenCLEnvironment().name
  ])
  config.opencl_opt.extend([True, False])
  harness = cl_launcher.ClLauncherHarness(config)
  assert len(harness.testbeds) == 2
  assert harness.testbeds[0].name == env.OclgrindOpenCLEnvironment().name
  assert harness.testbeds[0].opts['opencl_opt'] == 'enabled'
  assert harness.testbeds[1].name == env.OclgrindOpenCLEnvironment().name
  assert harness.testbeds[1].opts['opencl_opt'] == 'disabled'


def test_ClLauncherHarness_RunTestcases_no_testbed():
  """Test that invalid request params returned if no testbed requested."""
  config = harness_pb2.ClLauncherHarness()
  harness = cl_launcher.ClLauncherHarness(config)
  req = harness_pb2.RunTestcasesRequest(testbed=None, testcases=[])
  res = harness.RunTestcases(req, None)
  assert (res.status.returncode ==
          service_pb2.ServiceStatus.INVALID_REQUEST_PARAMETERS)
  assert res.status.error_message == 'Requested testbed not found.'


def test_ClLauncherHarness_RunTestcases_no_testcases():
  """Test that empty results returned if no testcase requested."""
  config = harness_pb2.ClLauncherHarness()
  harness = cl_launcher.ClLauncherHarness(config)
  assert len(harness.testbeds)
  req = harness_pb2.RunTestcasesRequest(testbed=harness.testbeds[0],
                                        testcases=[])
  res = harness.RunTestcases(req, None)
  assert res.status.returncode == service_pb2.ServiceStatus.SUCCESS
  assert not res.results


def test_ClLauncherHarness_RunTestcases_oclgrind_abc_testcase(
    abc_harness, abc_run_testcases_request):
  """And end-to-end test of the abc_testcase on oclgrind."""
  res = abc_harness.RunTestcases(abc_run_testcases_request, None)
  assert res.status.returncode == service_pb2.ServiceStatus.SUCCESS
  assert len(res.results) == 1
  result = res.results[0]

  # The returned testcase is identical to the input testcase.
  assert result.testcase == abc_run_testcases_request.testcases[0]

  # Check the result properties.
  assert result.outcome == deepsmith_pb2.Result.PASS
  print(result.outputs['stderr'])
  assert '3-D global size 1 = [1, 1, 1]' in result.outputs['stderr']
  assert '3-D local size 1 = [1, 1, 1]' in result.outputs['stderr']
  assert 'OpenCL optimizations: on' in result.outputs['stderr']
  assert 'Platform: ' in result.outputs['stderr']
  assert 'Device: ' in result.outputs['stderr']
  assert 'Compilation terminated successfully...'
  assert result.outputs['stdout'] == '0,'


def test_ClLauncherHarness_RunTestcases_oclgrind_syntax_error(
    abc_harness, abc_run_testcases_request):
  """Test outcome of kernel with syntax error."""
  abc_run_testcases_request.testcases[0].inputs['src'] = '!!11@invalid syntax'
  res = abc_harness.RunTestcases(abc_run_testcases_request, None)
  assert res.status.returncode == service_pb2.ServiceStatus.SUCCESS
  assert len(res.results) == 1
  result = res.results[0]
  assert result.outcome == deepsmith_pb2.Result.BUILD_FAILURE


if __name__ == '__main__':
  test.Main()
