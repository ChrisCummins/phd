"""Unit tests for //deeplearning/clgen/preprocessors/opencl.py."""
import sys

import pytest
from absl import app
from absl import flags
from absl import logging

import deeplearning.clgen
from deeplearning.clgen import errors, native
from deeplearning.clgen.preprocessors import opencl
from deeplearning.clgen.proto import corpus_pb2
from deeplearning.clgen.tests import testlib as tests


FLAGS = flags.FLAGS


def test_GetClangArgs_use_shim():
  args = opencl.GetClangArgs(use_shim=False)
  assert native.SHIMFILE not in args
  args = opencl.GetClangArgs()  # Default value is used.
  assert native.SHIMFILE not in args
  args = opencl.GetClangArgs(use_shim=True)
  assert native.SHIMFILE in args


@pytest.mark.skip(reason='TODO(cec) New preprocessor pipeline')
def test_compiler_preprocess_cl_no_change():
  """Test that code without preprocessor directives is unchanged."""
  src = "kernel void A(global int*a ) {}"
  assert opencl.ClangPreprocess(src) == src


@pytest.mark.skip(reason='TODO(cec) New preprocessor pipeline')
def test_compiler_preprocess_cl_whitespace():
  """Test that preprocessor output produces exactly one terminating newling."""
  src = "kernel void A(global int*a ) {}"
  # Leading whitespace is stripped.
  assert opencl.ClangPreprocess('\n\n' + src) == src
  # Trailing whitespace is stripped.
  assert opencl.ClangPreprocess(src + '\n\n') == src


@pytest.mark.skip(reason='TODO(cec) New preprocessor pipeline')
def test_compiler_preprocess_cl_user_directives():
  """Test inlining of user-defined preprocessor directives."""
  src = """\
#define MY_TYPE int
kernel void A(global MY_TYPE* a) {}
#ifdef SOME_CONDITIONAL_THING
kernel void B() {}
#endif
"""
  out = "kernel void A(global int* a) {}"
  assert opencl.ClangPreprocess(src) == out


@pytest.mark.skip(reason='TODO(cec) New preprocessor pipeline')
def test_compiler_preprocess_cl_undefined_macro():
  """Test that code with undefined macro is unchanged."""
  src = "kernel void A(global MY_TYPE* a) {}"
  assert opencl.ClangPreprocess(src) == src


@pytest.mark.skip(reason='TODO(cec) New preprocessor pipeline')
def test_preprocess_shim():
  """Test that code which contains defs in opencl-shim can compile."""
  # FLOAT_T is defined in shim header. Preprocess will fail if FLOAT_T is
  # undefined.
  preprocessors = [
    corpus_pb2.Preprocessor(name="opencl:ClangFrontendPreprocess")]
  with pytest.raises(errors.BadCodeException):
    preprocessors.preprocess("""
__kernel void A(__global FLOAT_T* a) { int b; }""", use_shim=False)

  assert preprocessors.preprocess("""
__kernel void A(__global FLOAT_T* a) { int b; }""", use_shim=True)


@pytest.mark.skip(reason='TODO(cec) New preprocessor pipeline')
def test_ugly_preprocessed():
  # empty kernel protoype is rejected
  with pytest.raises(errors.NoCodeException):
    preprocessors.preprocess("""\
__kernel void A() {
}\
""")
  # kernel containing some code returns the same.
  assert """\
__kernel void A() {
  int a;
}\
""" == preprocessors.preprocess("""\
__kernel void A() {
  int a;
}\
""")


@pytest.mark.skip(reason='TODO(cec) New preprocessor pipeline')
def test_preprocess_stable():
  code = """\
__kernel void A(__global float* a) {
  int b;
  float c;
  int d = get_global_id(0);

  a[d] *= 2.0f;
}\
"""
  # pre-processing is "stable" if the code doesn't change
  out = code
  for _ in range(5):
    out = preprocessors.preprocess(out)
    assert out == code


@pytest.mark.skip(reason='TODO(cec) New preprocessor pipeline')
@tests.needs_linux  # FIXME: GPUVerify support on macOS.
def test_gpuverify():
  code = """\
__kernel void A(__global float* a) {
  int b = get_global_id(0);
  a[b] *= 2.0f;
}"""
  assert opencl.GpuVerify(code, ["--local_size=64", "--num_groups=128"]) == code


@pytest.mark.skip(reason='TODO(cec) New preprocessor pipeline')
@tests.needs_linux  # FIXME: GPUVerify support on macOS.
def test_gpuverify_data_race():
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
