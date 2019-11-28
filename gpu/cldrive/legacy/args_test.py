# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
# This file is part of cldrive.
#
# cldrive is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cldrive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //gpu/cldrive/legacy/args.py."""
import pytest

from gpu.cldrive.legacy import args
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

# GetKernelArguments() tests.


def test_GetKernelArguments_hello_world():
  """Simple hello world argument type test."""
  args_ = args.GetKernelArguments("kernel void a(global float* a) {}")
  assert args_[0].typename == 'float'


def test_GetKernelArguments_proper_kernel():
  """Test kernel arguments of a "real" kernel."""
  args_ = args.GetKernelArguments("""
typedef int foobar;

void B(const int e);

__kernel void A(const __global int* data, __local float4 * restrict car,
                __global const float* b, const int foo, int d) {
  int tid = get_global_id(0);
  data[tid] *= 2.0;
}

void B(const int e) {}
""")
  assert len(args_) == 5
  assert args_[0].is_const
  assert args_[0].is_pointer
  assert args_[0].typename == "int"
  assert args_[0].bare_type == "int"


def test_GetKernelArguments_no_definition():
  """Test that error is raised if no kernel defined."""
  with pytest.raises(args.NoKernelError):
    args.GetKernelArguments("")


def test_GetKernelArguments_declared_but_not_defined():
  """Test that error is raised if kernel declared but not defined."""
  with pytest.raises(args.NoKernelError):
    args.GetKernelArguments("kernel void A();")


def test_GetKernelArguments_multiple_kernels():
  """Test that error is raised if no kernel defined."""
  with pytest.raises(args.MultipleKernelsError):
    args.GetKernelArguments("""
kernel void A() {}
kernel void B() {}
""")


def test_GetKernelArguments_struct_not_supported():
  """Test that error is raised if type is not supported."""
  with pytest.raises(ValueError) as e_ctx:
    args.GetKernelArguments("struct C; kernel void A(struct C a) {}")
  assert "Unsupported data type for argument: 'a'" == str(e_ctx.value)


def test_GetKernelArguments_local_global_qualified():
  """Test that error is raised if address space is invalid."""
  with pytest.raises(args.OpenCLValueError) as e_ctx:
    args.GetKernelArguments("kernel void A(global local int* a) {}")
  assert ("Pointer argument 'global local int *a' has multiple "
          'address space qualifiers') == str(e_ctx.value)


def test_GetKernelArguments_no_qualifiers():
  """Test that error is raised if argument has no address space qualifier."""
  with pytest.raises(args.OpenCLValueError) as e_ctx:
    args.GetKernelArguments("kernel void A(float* a) {}")
  assert "Pointer argument 'float *a' has no address space qualifier" == str(
      e_ctx.value)


def test_GetKernelArguments_no_args():
  """Test that no arguments returned for kernel with no args."""
  assert len(args.GetKernelArguments("kernel void A() {}")) == 0


def test_GetKernelArguments_address_spaces():
  """Test address space types."""
  args_ = args.GetKernelArguments("""
kernel void A(global int* a,
              local int* b,
              constant int* c,
              const int d) {}
""")
  assert len(args_) == 4
  assert args_[0].address_space == "global"
  assert args_[1].address_space == "local"
  assert args_[2].address_space == "constant"
  assert args_[3].address_space == "private"


def test_GetKernelArguments_properties():
  """Test extracted properties of kernel."""
  args_ = args.GetKernelArguments("""
kernel void A(const global int* a, global const float* b,
              local float4 *const c, const int d, float2 e) {}
""")
  assert len(args_) == 5
  assert args_[0].is_pointer
  assert args_[0].address_space == "global"
  assert args_[0].typename == "int"
  assert args_[0].name == "a"
  assert args_[0].bare_type == "int"
  assert not args_[0].is_vector
  assert args_[0].vector_width == 1
  assert args_[0].is_const

  assert args_[1].is_pointer
  assert args_[1].address_space == "global"
  assert args_[1].typename == "float"
  assert args_[1].name == "b"
  assert args_[1].bare_type == "float"
  assert not args_[1].is_vector
  assert args_[1].vector_width == 1
  assert args_[1].is_const

  assert args_[2].is_pointer
  assert args_[2].address_space == "local"
  assert args_[2].typename == "float4"
  assert args_[2].name == "c"
  assert args_[2].bare_type == "float"
  assert args_[2].is_vector
  assert args_[2].vector_width == 4
  assert not args_[2].is_const


# ParseSource() tests.


def test_ParseSource_hello_world():
  """Test simple kernel parse."""
  ast = args.ParseSource("kernel void A() {}")
  assert len(ast.children()) >= 1


def test_ParseSource_syntax_error():
  """Test that error is raised if source contains error."""
  src = "kernel void A(@!"
  with pytest.raises(args.OpenCLValueError) as e_ctx:
    args.ParseSource(src)
  assert "Syntax error: ':1:15: Illegal character '@''" == str(e_ctx.value)

  # OpenCLValueError extends ValueError.
  with pytest.raises(ValueError):
    args.ParseSource(src)


# GetKernelName() tests.


def test_GetKernelName_hello_world():
  """Test name extraction from a simple kernel."""
  assert 'hello_world' == args.GetKernelName('kernel void hello_world() {}')


def test_GetKernelName_hello_world_with_distractions():
  """A simple kernel test with unusual formatting."""
  assert 'hello_world' == args.GetKernelName("""
int A() {}

kernel
  void
hello_world(global int* a, const int c) {
  a[0] = 0;
}
""")


def test_GetKernelName_no_kernels():
  """Test that error is raised if no kernels are defined."""
  with pytest.raises(args.NoKernelError) as e_ctx:
    args.GetKernelName('')
  assert 'Source contains no kernel definitions' == str(e_ctx.value)

  with pytest.raises(args.NoKernelError) as e_ctx:
    args.GetKernelName('int A() {}')
  assert 'Source contains no kernel definitions' == str(e_ctx.value)


def test_GetKernelName_multiple_kernels():
  """Test that error is raised if multiple kernels are defined."""
  with pytest.raises(args.MultipleKernelsError) as e_ctx:
    args.GetKernelName("""
kernel void A() {}
kernel void B() {}
""")
  assert 'Source contains more than one kernel definition' == str(e_ctx.value)


def test_GetKernelName_syntax_error():
  """Test that error is raised if source contains syntax error."""
  with pytest.raises(args.OpenCLValueError) as e_ctx:
    args.GetKernelName('!@##syntax error!!!!1')
  assert "Syntax error: ':1:1: before: !'" == str(e_ctx.value)


if __name__ == "__main__":
  test.Main()
