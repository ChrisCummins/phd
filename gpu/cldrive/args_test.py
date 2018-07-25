"""Unit tests for //gpu/cldrive/args.py."""
import pytest
import sys
from absl import app

from gpu.cldrive import args


# @pytest.mark.skip(reason="FIXME(cec)")
# def test_preprocess():
#   pp = args.PreprocessSource("kernel void A() {}")
#   assert pp.split("\n")[-2] == "kernel void A() {}"
#
#
# def test_parse():
#   ast = args.ParseSource("kernel void A() {}")
#   assert len(ast.children()) >= 1
#
#
# def test_extract_args_syntax_error():
#   src = "kernel void A(@!"
#   with pytest.raises(args.OpenCLValueError):
#     args.ParseSource(src)
#
#   # OpenCLValueError extends ValueError
#   with pytest.raises(ValueError):
#     args.ParseSource(src)
#
#
# def test_parse_preprocess():
#   src = """
#     #define DTYPE float
#     kernel void A(global DTYPE *a) {}
#     """
#
#   pp = args.PreprocessSource(src)
#   ast = args.ParseSource(pp)
#   assert len(ast.children()) >= 1
#
#
# @pytest.mark.skip(reason="FIXME(cec)")
# def test_parse_header():
#   src = """
#     #include "header.h"
#     kernel void A(global DTYPE* a) {
#       a[get_global_id(0)] = DOUBLE(a[get_global_id(0)]);
#     }
#     """
#   pp = args.PreprocessSource(src, include_dirs=[data_path("")])
#   ast = args.ParseSource(pp)
#   assert len(ast.children()) >= 1
#
#
# def test_extract_args():
#   src = """
#     typedef int foobar;
#
#     void B(const int e);
#
#     __kernel void A(const __global int* data, __local float4 * restrict car,
#                     __global const float* b, const int foo, int d) {
#         int tid = get_global_id(0);
#         data[tid] *= 2.0;
#     }
#
#     void B(const int e) {}
#     """
#   args = args.GetKernelArguments(src)
#
#   assert len(args) == 5
#   assert args[0].is_const
#   assert args[0].is_pointer
#   assert args[0].typename == "int"
#   assert args[0].bare_type == "int"
#
#
# def test_extract_args_no_declaration():
#   with pytest.raises(LookupError):
#     args.GetKernelArguments("")
#
#
# def test_extract_args_no_definition():
#   src = "kernel void A();"
#   with pytest.raises(LookupError):
#     args.GetKernelArguments(src)
#
#
# def test_extract_args_multiple_kernels():
#   src = "kernel void A() {} kernel void B() {}"
#   with pytest.raises(LookupError):
#     args.GetKernelArguments(src)
#
#
# def test_extract_args_struct():
#   src = "struct C; kernel void A(struct C a) {}"
#   with pytest.raises(ValueError):
#     args.GetKernelArguments(src)
#
#
# def test_extract_args_local_global_qualified():
#   src = "kernel void A(global local int* a) {}"
#   with pytest.raises(args.OpenCLValueError):
#     args.GetKernelArguments(src)
#
#
# def test_extract_args_no_qualifiers():
#   src = "kernel void A(float* a) {}"
#   with pytest.raises(args.OpenCLValueError):
#     args.GetKernelArguments(src)
#
#
# def test_extract_args_no_args():
#   src = "kernel void A() {}"
#   assert len(args.GetKernelArguments(src)) == 0
#
#
# def test_extract_args_address_spaces():
#   src = """
#     kernel void A(global int* a, local int* b, constant int* c, const int d) {}
#     """
#   args = args.GetKernelArguments(src)
#   assert len(args) == 4
#   assert args[0].address_space == "global"
#   assert args[1].address_space == "local"
#   assert args[2].address_space == "constant"
#   assert args[3].address_space == "private"
#
#
# def test_extract_args_no_address_space():
#   src = """
#     kernel void A(int* a) {}
#     """
#   with pytest.raises(args.OpenCLValueError):
#     args = args.GetKernelArguments(src)
#
#
# def test_extract_args_multiple_address_spaces():
#   src = """
#     kernel void A(global local int* a) {}
#     """
#   with pytest.raises(args.OpenCLValueError):
#     args = args.GetKernelArguments(src)
#
#
# def test_extract_args_properties():
#   src = """
#     kernel void A(const global int* a, global const float* b,
#                   local float4 *const c, const int d, float2 e) {}
#     """
#   args = args.GetKernelArguments(src)
#   assert len(args) == 5
#   assert args[0].is_pointer == True
#   assert args[0].address_space == "global"
#   assert args[0].typename == "int"
#   assert args[0].name == "a"
#   assert args[0].bare_type == "int"
#   assert args[0].is_vector == False
#   assert args[0].vector_width == 1
#   assert args[0].is_const == True
#
#   assert args[1].is_pointer == True
#   assert args[1].address_space == "global"
#   assert args[1].typename == "float"
#   assert args[1].name == "b"
#   assert args[1].bare_type == "float"
#   assert args[1].is_vector == False
#   assert args[1].vector_width == 1
#   assert args[1].is_const == True
#
#   assert args[2].is_pointer == True
#   assert args[2].address_space == "local"
#   assert args[2].typename == "float4"
#   assert args[2].name == "c"
#   assert args[2].bare_type == "float"
#   assert args[2].is_vector == True
#   assert args[2].vector_width == 4
#   assert args[2].is_const == False
#
#
# def test_extract_args_struct():
#   src = """
#     struct s { int a; };
#
#     kernel void A(global struct s *a) {}
#     """
#   # we can't handle structs yet
#   with pytest.raises(args.OpenCLValueError):
#     args.GetKernelArguments(src)
#
#
# def test_extract_args_preprocess():
#   src = """
#     #define DTYPE float
#     kernel void A(global DTYPE *a) {}
#     """
#
#   pp = args.PreprocessSource(src)
#   args = args.GetKernelArguments(pp)
#   assert args[0].typename == 'float'


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


def test_GetKernelName_multiple_kernels():
  """Test that error is raised if multiple kernels are defined."""
  with pytest.raises(args.MultipleKernelsError) as e_ctx:
    args.GetKernelName("""
kernel void A() {}
kernel void B() {}  
""")
  assert 'Source contains more than one kernel definition' == str(e_ctx.value)


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main(
      [args.__file__, __file__, "-vv", "--doctest-modules"]))


if __name__ == "__main__":
  app.run(main)
