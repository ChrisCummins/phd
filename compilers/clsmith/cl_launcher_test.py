"""Unit tests for //compilers/clsmith/cl_launcher.py."""

from absl import flags

from compilers.clsmith import cl_launcher
from gpu.cldrive.legacy import driver
from gpu.cldrive.legacy import env
from labm8 import test

FLAGS = flags.FLAGS

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


def test_ExecClsmithSource_pass():
  """And end-to-end test of executing a CLSmith source."""
  env_ = env.OclgrindOpenCLEnvironment()
  proc = cl_launcher.ExecClsmithSource(env_, CLSMITH_EXAMPLE_SRC,
                                       driver.NDRange(1, 1, 1),
                                       driver.NDRange(1, 1, 1), '---debug')

  assert not proc.returncode
  assert '3-D global size 1 = [1, 1, 1]' in proc.stderr
  assert '3-D local size 1 = [1, 1, 1]' in proc.stderr
  assert 'OpenCL optimizations: on' in proc.stderr
  assert 'Platform: ' in proc.stderr
  assert 'Device: ' in proc.stderr
  assert 'Compilation terminated successfully...'
  assert proc.stdout == '0,'


def test_ExecClsmithSource_syntax_error():
  """Test outcome of kernel with syntax error."""
  env_ = env.OclgrindOpenCLEnvironment()
  proc = cl_launcher.ExecClsmithSource(env_, "!@!###syntax error!",
                                       driver.NDRange(1, 1, 1),
                                       driver.NDRange(1, 1, 1), '---debug')

  assert proc.returncode == 1
  assert proc.stdout == ''
  assert 'Error building program: -11' in proc.stderr


if __name__ == '__main__':
  test.Main()
