"""Unit tests for //deeplearning/clgen/preprocessors/clang.py."""
import subprocess
import sys

import pytest
from absl import app
from absl import flags

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import clang


FLAGS = flags.FLAGS


class MockProcess():
  """Mock class for subprocess.Popen() return."""

  def __init__(self, returncode):
    self.returncode = returncode

  def communicate(self, input):
    return '', ''


# StripPreprocessorLines() tests.

def test_StripPreprocessorLines_empty_input():
  assert clang.StripPreprocessorLines('') == ''


def test_StripPreprocessorLines_no_stdin():
  """Test that without a stdin marker is stripped."""
  assert clang.StripPreprocessorLines('The\ncat\nsat\non\nthe\nmat') == ''


def test_StripPreprocessorLines_everything_stripped_before_stdin():
  """Test that everything before the stdin marker is stripped."""
  assert clang.StripPreprocessorLines("""
This will
all be stripped.
# 1 "<stdin>" 2
This will not be stripped.""") == "This will not be stripped."


def test_StripPreprocessorLines_nothing_before_stdin():
  """Test that if nothing exists before stdin marker, nothing is lost."""
  assert clang.StripPreprocessorLines('# 1 "<stdin>" 2\nfoo') == 'foo'


def test_StripPreprocessorLines_hash_stripped():
  """Test that lines which begin with # symbol are stripped."""
  assert clang.StripPreprocessorLines("""# 1 "<stdin>" 2
# This will be stripped.
Foo
This will # not be stripped.
# But this will.""") == """Foo
This will # not be stripped."""


# Preprocess() tests.

def test_Preprocess_process_command(mocker):
  """Test the process comand which is run."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(0)
  clang.Preprocess('', ['-foo'])
  subprocess.Popen.assert_called_once()
  cmd = subprocess.Popen.call_args_list[0][0][0]
  assert cmd[:3] == ['timeout', '-s9', '60']
  assert cmd[4:] == ['-E', '-c', '-', '-o', '-', '-foo']


def test_Preprocess_ClangTimeout(mocker):
  """Test that ClangTimeout is raised if clang returns with SIGKILL."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(9)
  with pytest.raises(errors.ClangTimeout):
    clang.Preprocess('', [])
  # ClangTimeout inherits from ClangException.
  with pytest.raises(errors.ClangException):
    clang.Preprocess('', [])


def test_Preprocess_ClangException(mocker):
  """Test that ClangException is raised if clang returns non-zero returncode."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(1)
  with pytest.raises(errors.ClangException):
    clang.Preprocess('', [])


def test_Preprocess_empty_file():
  """Test the preprocessor output with an empty file."""
  assert clang.Preprocess('', []) == '\n'


def test_Preprocess_simple_c_program():
  """Test that a simple C program is unchanged."""
  assert clang.Preprocess("""
int main(int argc, char** argv) { return 0; }
""", []) == """
int main(int argc, char** argv) { return 0; }
"""


def test_Preprocess_inlined_cflag():
  """Test pre-processing with a custom define in the command line."""
  assert clang.Preprocess("""
int main(MY_TYPE argc, char** argv) { return 0; }
""", ['-DMY_TYPE=int']) == """
int main(int argc, char** argv) { return 0; }
"""


def test_Preprocess_inlined_define():
  """Test pre-processing with a #define in the source code."""
  assert clang.Preprocess("""
#define MY_TYPE int
int main(MY_TYPE argc, char** argv) { return 0; }
""", ['-DMY_TYPE=int']) == """

int main(int argc, char** argv) { return 0; }
"""


def test_Preprocess_undefined_data_type():
  """Test that an undefined data type does not cause an error."""
  assert clang.Preprocess("""
int main(MY_TYPE argc, char** argv) { return 0; }
""", []) == """
int main(MY_TYPE argc, char** argv) { return 0; }
"""


def test_Preprocess_undefined_variable():
  """Test that an undefined variable does not cause an error."""
  assert clang.Preprocess("""
int main(int argc, char** argv) { return UNDEFINED_VARIABLE; }
""", []) == """
int main(int argc, char** argv) { return UNDEFINED_VARIABLE; }
"""


def test_Preprocess_undefined_function():
  """Test that an undefined function does not cause an error."""
  assert clang.Preprocess("""
int main(int argc, char** argv) { return UNDEFINED_FUNCTION(0); }
""", []) == """
int main(int argc, char** argv) { return UNDEFINED_FUNCTION(0); }
"""


def test_Preprocess_invalid_preprocessor_directive():
  """Test that an invalid preprocessor directive raises an error."""
  with pytest.raises(errors.ClangException) as e_info:
    clang.Preprocess("""
#this_is_not_a_valid_directive
int main(int argc, char** argv) { return 0; }
""", [])
  assert "invalid preprocessing directive" in str(e_info.value)


def test_Preprocess_no_strip_processor_lines():
  """Test that stdin marker is preserved if no strip_preprocessor_lines."""
  assert '# 1 "<stdin>" 2' in clang.Preprocess("""
int main(int argc, char** argv) { return 0; }
""", [], strip_preprocessor_lines=False)


def test_Preprocess_include_stdio_strip():
  """Test that an included file is stripped."""
  assert clang.Preprocess("""
#include <stdio.h>
int main(int argc, char** argv) { return NULL; }
""", []) == """\
int main(int argc, char** argv) { return ((void *)0); }
"""


# ClangFormat() tests.

def test_ClangFormat_process_command(mocker):
  """Test the clang-format comand which is run."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(0)
  clang.ClangFormat('')
  subprocess.Popen.assert_called_once()
  cmd = subprocess.Popen.call_args_list[0][0][0]
  assert cmd[:3] == ['timeout', '-s9', '60']


def test_ClangFormat_ClangTimeout(mocker):
  """Test that ClangTimeout is raised if clang-format returns with SIGKILL."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(9)
  with pytest.raises(errors.ClangTimeout):
    clang.ClangFormat('')
  # ClangTimeout inherits from ClangException.
  with pytest.raises(errors.ClangException):
    clang.ClangFormat('')


def test_ClangFormat_ClangException(mocker):
  """Test that ClangException is raised if clang-format returns non-zero."""
  mock_Popen = mocker.patch('subprocess.Popen')
  mock_Popen.return_value = MockProcess(1)
  with pytest.raises(errors.ClangFormatException):
    clang.ClangFormat('')


def test_ClangFormat_empty_file():
  """Test the preprocessor output with an empty file."""
  assert clang.ClangFormat('') == ''


def test_ClangFormat_simple_c_program():
  """Test that a simple C program is unchanged."""
  assert clang.ClangFormat("""
int main(int argc, char** argv) { return 0; }
""") == """
int main(int argc, char** argv) {
  return 0;
}
"""


def test_ClangFormat_pointer_alignment():
  """Test that pointers are positioned left."""
  assert clang.ClangFormat("""
int * A(int* a, int * b, int *c);
""") == """
int* A(int* a, int* b, int* c);
"""


def test_ClangFormat_undefined_data_type():
  """Test that an undefined data type does not cause an error."""
  assert clang.ClangFormat("""
int main(MY_TYPE argc, char** argv) { return 0; }
""") == """
int main(MY_TYPE argc, char** argv) {
  return 0;
}
"""


def test_ClangFormat_undefined_variable():
  """Test that an undefined variable does not cause an error."""
  assert clang.ClangFormat("""
int main(int argc, char** argv) { return UNDEFINED_VARIABLE; }
""") == """
int main(int argc, char** argv) {
  return UNDEFINED_VARIABLE;
}
"""


def test_ClangFormat_undefined_function():
  """Test that an undefined function does not cause an error."""
  assert clang.ClangFormat("""
int main(int argc, char** argv) { return UNDEFINED_FUNCTION(0); }
""") == """
int main(int argc, char** argv) {
  return UNDEFINED_FUNCTION(0);
}
"""


def test_ClangFormat_invalid_preprocessor_directive():
  """Test that an invalid preprocessor directive does not raise an error."""
  assert clang.ClangFormat("""
#this_is_not_a_valid_directive
int main(int argc, char** argv) { return 0; }
""") == """
#this_is_not_a_valid_directive
int main(int argc, char** argv) {
  return 0;
}
"""


@pytest.mark.skip(reason='TODO(cec): Bytecode compiler')
def test_CompilerLlvmBytecode_empty_file():
  """Test that bytecode is produced for good code."""
  assert not clang.CompileLlvmBytecode("int main() {}", [])


@pytest.mark.skip(reason='TODO(cec): Bytecode compiler')
def test_compile_cl_bytecode_undefined_type():
  """Test that error is raised when kernel contains undefined type."""
  with pytest.raises(errors.ClangException):
    clang.CompileLlvmBytecode("kernel void A(global FLOAT_T* a) {}", "<anon>",
                              use_shim=False)


@pytest.mark.skip(reason='TODO(cec): Bytecode compiler')
def test_compile_cl_bytecode_shim_type():
  """Test that bytecode is produced for code with shim type."""
  assert clang.CompileLlvmBytecode("kernel void A(global FLOAT_T* a) {}",
                                   "<anon>", use_shim=True)


@pytest.mark.skip(reason="TODO(cec): Bytecode compiler")
def test_bytecode_features_empty_code():
  # Generated by CompileLlvmBytecode from: 'kernel void A(global float* a) {}'.
  bc = """\
; ModuleID = '-'
source_filename = "-"
target datalayout = "e-i64:64-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-nvcl"

; Function Attrs: noinline norecurse nounwind readnone
define spir_kernel void @A(float addrspace(1)* nocapture) local_unnamed_addr 
#0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 
!kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
  ret void
}

attributes #0 = { noinline norecurse nounwind readnone 
"correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" 
"less-precise-fpmad"="false" "no-frame-pointer-elim"="true" 
"no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" 
"no-jump-tables"="false" "no-nans-fp-math"="false" 
"no-signed-zeros-fp-math"="false" "no-trapping-math"="false" 
"stack-protector-buffer-size"="8" "target-features"="-satom" 
"unsafe-fp-math"="false" "use-soft-float"="false" }

!nvvm.annotations = !{!0}
!llvm.module.flags = !{!1}
!opencl.ocl.version = !{!2}
!llvm.ident = !{!3}

!0 = !{void (float addrspace(1)*)* @A, !"kernel", i32 1}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 1, i32 0}
!3 = !{!"clang version 5.0.0 (tags/RELEASE_500/final)"}
!4 = !{i32 1}
!5 = !{!"none"}
!6 = !{!"float*"}
!7 = !{!""}
"""
  assert clang.bytecode_features(bc, "<anon>")


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
