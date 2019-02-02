"""Unit tests for //deeplearning/clgen/preprocessors/opencl.py."""
import subprocess
import sys

import pytest
from absl import app
from absl import flags
from absl import logging

import deeplearning.clgen
from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import opencl
from labm8 import bazelutil


FLAGS = flags.FLAGS

SHIMFILE = bazelutil.DataPath(
    'phd/deeplearning/clgen/data/include/opencl-shim.h')


class MockProcess(object):
  """Mock class for subprocess.Popen() return."""

  def __init__(self, returncode):
    self.returncode = returncode

  def communicate(self, *args):
    del args
    return '', ''


# GetClangArgs() tests.

def test_GetClangArgs_no_shim():
  args = opencl.GetClangArgs(use_shim=False)
  assert str(SHIMFILE) not in args


def test_GetClangArgs_with_shim():
  args = opencl.GetClangArgs(use_shim=True)
  assert str(SHIMFILE) in args


# ClangPreprocess() tests.

def test_ClangPreprocess_small_program():
  """Test that a small program without preprocessor directives is unchanged."""
  assert opencl.ClangPreprocess("""
kernel void A(global int*a ) {}
""") == """
kernel void A(global int*a ) {}
"""


def test_ClangPreprocess_user_directives():
  """Test inlining of user-defined preprocessor directives."""
  assert opencl.ClangPreprocess("""
#define MY_TYPE int
kernel void A(global MY_TYPE* a) {}
#ifdef SOME_CONDITIONAL_THING
kernel void B() {}
#endif
""") == """

kernel void A(global int* a) {}
"""


def test_ClangPreprocess_undefined_macro():
  """Test that code with undefined macro is unchanged."""
  assert opencl.ClangPreprocess("""
kernel void A(global FLOAT_T* a) {}
""") == """
kernel void A(global FLOAT_T* a) {}
"""


# ClangPreprocessWithShim() tests.

def test_ClangPreprocessWithShim_compiler_args(mocker):
  """Test that shimfile is in comand which is run."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(0)
  opencl.ClangPreprocessWithShim('')
  subprocess.Popen.assert_called_once()
  cmd = subprocess.Popen.call_args_list[0][0][0]
  assert str(SHIMFILE) in cmd


def test_ClangPreprocessWithShim_shim_define():
  """Test that code which contains defs in opencl-shim can compile."""
  # FLOAT_T is defined in shim header. Preprocess will fail if FLOAT_T is
  # undefined.
  assert opencl.ClangPreprocessWithShim("""
kernel void A(global FLOAT_T* a) {}
""") == """
kernel void A(global float* a) {}
"""


# Compile() tests.

def test_Compile_empty_input():
  """Test that Compile accepts an empty input."""
  assert opencl.Compile('') == ''


def test_Compile_small_program():
  """Test that Compile accepts a small program."""
  assert opencl.Compile("""
kernel void A(global int*a ) {
  a[get_global_id(0)] = 0;
}
""") == """
kernel void A(global int*a ) {
  a[get_global_id(0)] = 0;
}
"""


def test_Compile_missing_shim_define():
  """Test that Compile rejects a program which depends on the shim header."""
  with pytest.raises(errors.ClangException):
    opencl.Compile("""
kernel void A(global FLOAT_T* a) {}
""")


def test_Compile_user_define():
  """Test that Compile accepts a program with a custom #define."""
  assert opencl.Compile("""
#define FLOAT_T float
kernel void A(global FLOAT_T* a) {}
""") == """
#define FLOAT_T float
kernel void A(global FLOAT_T* a) {}
"""


def test_Compile_syntax_error():
  """Test that Compile rejects a program with invalid syntax."""
  with pytest.raises(errors.ClangException) as e_info:
    opencl.Compile("kernel void A2@@1!!!#")
  assert 'error: ' in str(e_info.value)


def test_Compile_undefined_variable():
  """Test that Compile rejects a program with an undefined variable."""
  with pytest.raises(errors.ClangException) as e_info:
    opencl.Compile("""
kernel void A(global int* a) {
  undefined_variable;
}
""")
  assert 'use of undeclared identifier' in str(e_info.value)


def test_Compile_undefined_function():
  """Test that Compile rejects a program with an undefined function."""
  with pytest.raises(errors.ClangException) as e_info:
    opencl.Compile("""
kernel void A(global int* a) {
  undefined_function(a);
}
""")
  assert 'implicit declaration of function' in str(e_info.value)


# NormalizeIdentifiers() tests.

def test_NormalizeIdentifiers_small_opencl_program():
  """Test that rewriter performs as expected for a small OpenCL program."""
  assert """
void kernel A(global int* a) {
  int b = 0;
  a[get_global_id(0)] = b;
}
""" == opencl.NormalizeIdentifiers("""
void kernel foo(global int* bar) {
  int car = 0;
  bar[get_global_id(0)] = car;
}
""")


def test_NormalizeIdentifiers_global_variable():
  """Test that global variable is renamed."""
  assert opencl.NormalizeIdentifiers("""
sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
const int foo = 0;
""") == """
sampler_t Ga = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
const int Gb = 0;
"""


def test_NormalizeIdentifiers_global_variable_reference():
  """Test that global variable reference is renamed."""
  assert opencl.NormalizeIdentifiers("""
const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

kernel void A() {
  sampler_t a = sampler;
}
""") == """
const sampler_t Ga = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

kernel void A() {
  sampler_t a = Ga;
}
"""

@pytest.mark.xfail(
    reason='FIXME: Global sampler variable references are not visited.'
)
def test_NormalizeIdentifiers_opencl_global_sampler_reference():
  """Test that global sampler variable references are rewritten."""
  assert opencl.NormalizeIdentifiers("""
constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
constant float2 vec = (float2)(0, 0);

float interpolate(read_only image2d_t image) {
  return read_imageui(image, sampler, vec).x;
}
""") == """
constant sampler_t Ga = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;
constant float2 Gb = (float2)(0, 0);

float A(read_only image2d_t a) {
  return read_imageui(a, Ga, Gb).x;
}
"""

@pytest.mark.xfail(
    reason='FIXME: Global sampler variable references are not visited.'
)
def test_NormalizeIdentifiers_opencl_regression_test():
  """An input file which triggered a rewrite failure in an earlier version."""
  assert opencl.NormalizeIdentifiers("""
constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

inline int interpolate(const float x, const float y, read_only image2d_t image) {
  const float ix = floor(x);
  const float dx = x - ix;

  const float iy = floor(y);
  const float dy = y - iy;

  const float intensity =
      read_imageui(image, sampler, (float2)(ix, iy)).x * (1 - dx) * (1 - dy)
      + read_imageui(image, sampler, (float2)(ix+1, iy)).x * dx * (1 - dy)
      + read_imageui(image, sampler, (float2)(ix, iy+1)).x * (1 - dx) * dy
      + read_imageui(image, sampler, (float2)(ix+1, iy+1)).x * dx * dy;

  return intensity;
}
""") == """
constant sampler_t Ga = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

inline int A(const float a, const float b, read_only image2d_t c) {
  const float d = floor(a);
  const float e = a - d;

  const float f = floor(b);
  const float g = b - f;

  const float h =
      read_imageui(c, Ga, (float2)(d, f)).x * (1 - e) * (1 - g)
      + read_imageui(c, Ga, (float2)(d+1, f)).x * e * (1 - g)
      + read_imageui(c, Ga, (float2)(d, f+1)).x * (1 - e) * g
      + read_imageui(c, Ga, (float2)(d+1, f+1)).x * e * g;

  return h;
}
"""


# SanitizeKernelPrototype() tests.

def test_SanitizeKernelPrototype_empty_input():
  """Test SanitizeKernelPrototype on an empty input."""
  assert opencl.SanitizeKernelPrototype('') == ''


def test_SanitizeKernelPrototype_leading_whitespace():
  """Test that SanitizeKernelPrototype strips leading whitespace."""
  assert opencl.SanitizeKernelPrototype("""
kernel void A(global float* a) {}
""") == """\
kernel void A(global float* a) {}
"""


def test_SanitizeKernelPrototype_multiple_spaces():
  """Test that SanitizeKernelPrototype removes double whitespace."""
  assert opencl.SanitizeKernelPrototype("""\
  kernel  void   A(global    float*  a) {}
""") == """\
kernel void A(global float* a) {}
"""


# StripDoubleUnderscorePrefixes() tests.

def test_StripDoubleUnderscorePrefixes_empty_input():
  assert opencl.StripDoubleUnderscorePrefixes('') == ''


def test_StripDoubleUnderscorePrefixes_simple_kernel():
  assert opencl.StripDoubleUnderscorePrefixes("""
__kernel void A(__global int* a) {
  __private int b;
}
""") == """
kernel void A(global int* a) {
  private int b;
}
"""


# Benchmarks.

HELLO_WORLD_CL = """
__kernel void A(__global int* a) {
  a[get_global_id(0)] = 0;
}
"""


def test_benchmark_ClangPreprocess_hello_world(benchmark):
  """Benchmark ClangPreprocess on a "hello world" OpenCL program."""
  benchmark(opencl.ClangPreprocess, HELLO_WORLD_CL)


def test_benchmark_ClangPreprocessWithShim_hello_world(benchmark):
  """Benchmark ClangPreprocessWithShim on a "hello world" OpenCL program."""
  benchmark(opencl.ClangPreprocessWithShim, HELLO_WORLD_CL)


def test_benchmark_Compile_hello_world(benchmark):
  """Benchmark Compile on a "hello world" OpenCL program."""
  benchmark(opencl.Compile, HELLO_WORLD_CL)


def test_benchmark_StripDoubleUnderscorePrefixes_hello_world(benchmark):
  """Benchmark StripDoubleUnderscorePrefixes on a "hello world" program."""
  benchmark(opencl.StripDoubleUnderscorePrefixes, HELLO_WORLD_CL)


@pytest.mark.skip(reason='TODO(cec): Re-enable GPUVerify support.')
def test_GpuVerify():
  code = """\
__kernel void A(__global float* a) {
  int b = get_global_id(0);
  a[b] *= 2.0f;
}"""
  assert opencl.GpuVerify(code, ["--local_size=64", "--num_groups=128"]) == code


@pytest.mark.skip(reason='TODO(cec): Re-enable GPUVerify support.')
def test_GpuVerify_data_race():
  code = """\
__kernel void A(__global float* a) {
  a[0] +=  1.0f;
}"""
  with pytest.raises(deeplearning.clgen.errors.GPUVerifyException):
    opencl.GpuVerify(code, ["--local_size=64", "--num_groups=128"])


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  logging.set_verbosity(logging.DEBUG)
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
