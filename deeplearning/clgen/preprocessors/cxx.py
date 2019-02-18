"""Preprocessor functions for C++."""
import re
import sys

from absl import flags

from compilers.llvm import clang as clanglib
from compilers.llvm import llvm
from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import clang
from deeplearning.clgen.preprocessors import normalizer
from deeplearning.clgen.preprocessors import public
from labm8 import bazelutil


FLAGS = flags.FLAGS

_UNAME = 'mac' if sys.platform == 'darwin' else 'linux'
LIBCXX_HEADERS = bazelutil.DataPath(f'libcxx_{_UNAME}/include/c++/v1')
CLANG_HEADERS = bazelutil.DataPath(f'libcxx_{_UNAME}/lib/clang/6.0.0/include')

C_COMMENT_RE = re.compile(
    r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
    re.DOTALL | re.MULTILINE)

# Flags to compile C++ files with. I've replicated the default search path,
# but substituted the sandboxed header locations in place of the defaults.
#   bazel-phd/bazel-out/*-py3-opt/bin/deeplearning/clgen/preprocessors/\
#     cxx_test.runfiles/llvm_mac/bin/clang -xc++ -E - -v
CLANG_ARGS = [
  '-xc++', '-isystem', str(LIBCXX_HEADERS), '-isystem', '/usr/local/include',
  '-isystem', str(CLANG_HEADERS), '-isystem', '/usr/include',
  '-Wno-ignored-pragmas', '-ferror-limit=1',
  '-Wno-implicit-function-declaration',
  '-Wno-incompatible-library-redeclaration', '-Wno-macro-redefined',
  '-Wno-unused-parameter', '-Wno-long-long', '-Wcovered-switch-default',
  '-Wdelete-non-virtual-dtor', '-Wstring-conversion',
  '-DLLVM_BUILD_GLOBAL_ISEL', '-D__STDC_CONSTANT_MACROS',
  '-D__STDC_FORMAT_MACROS', '-D__STDC_LIMIT_MACROS',
  '-D_LIBCPP_HAS_C_ATOMIC_IMP'
]


@public.clgen_preprocessor
def ClangPreprocess(text: str) -> str:
  try:
    return clang.StripPreprocessorLines(clanglib.Preprocess(text, CLANG_ARGS))
  except llvm.LlvmError as e:
    raise errors.ClangException(str(e.stderr[:1024]))


@public.clgen_preprocessor
def Compile(text: str) -> str:
  """A preprocessor which attempts to compile the given code.

  Args:
    text: Code to compile.

  Returns:
    The input code, unmodified.
  """
  clang.CompileLlvmBytecode(text, '.cpp', CLANG_ARGS)
  return text


@public.clgen_preprocessor
def ClangFormat(text: str) -> str:
  """Run clang-format on a source to enforce code style.

  Args:
    text: The source code to run through clang-format.

  Returns:
    The output of clang-format.

  Raises:
    ClangFormatException: In case of an error.
    ClangTimeout: If clang-format does not complete before timeout_seconds.
  """
  return clang.ClangFormat(text, '.cpp')


@public.clgen_preprocessor
def NormalizeIdentifiers(text: str) -> str:
  """Normalize identifiers in C++ source code.

  Args:
    text: The source code to rewrite.

  Returns:
    Source code with identifier names normalized.

  Raises:
    RewriterException: If rewriter found nothing to rewrite.
    ClangTimeout: If rewriter fails to complete within timeout_seconds.
  """
  return normalizer.NormalizeIdentifiers(text, '.cpp', CLANG_ARGS)


@public.clgen_preprocessor
def StripComments(text: str) -> str:
  """Strip C/C++ style comments.

  Written by @markus-jarderot https://stackoverflow.com/a/241506/1318051
  """

  def Replacer(match):
    """Regex replacement callback."""
    s = match.group(0)
    if s.startswith('/'):
      return " "  # note: a space and not an empty string
    else:
      return s

  return C_COMMENT_RE.sub(Replacer, text)
