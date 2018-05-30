"""Preprocessor functions for C++."""
import pathlib
import re

from config import getconfig
from deeplearning.clgen.preprocessors import clang
from deeplearning.clgen.preprocessors import normalizer
from deeplearning.clgen.preprocessors import public
from lib.labm8 import bazelutil


_config = getconfig.GetGlobalConfig()
if _config.uname == 'darwin':
  LIBCXX_HEADERS = pathlib.Path(_config.paths.llvm_prefix) / 'include/c++/v1'
else:
  LIBCXX_HEADERS = bazelutil.DataPath('libcxx/include')

C_COMMENT_RE = re.compile(
    r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
    re.DOTALL | re.MULTILINE)

CLANG_ARGS = ['-xc++', '-isystem', str(LIBCXX_HEADERS),
              '-Wno-ignored-pragmas', '-ferror-limit=1',
              '-Wno-implicit-function-declaration',
              '-Wno-incompatible-library-redeclaration', '-Wno-macro-redefined',
              '-Wno-unused-parameter', '-Wno-long-long',
              '-Wcovered-switch-default', '-Wdelete-non-virtual-dtor',
              '-Wstring-conversion', '-DLLVM_BUILD_GLOBAL_ISEL',
              '-D__STDC_CONSTANT_MACROS', '-D__STDC_FORMAT_MACROS',
              '-D__STDC_LIMIT_MACROS', '-D_LIBCPP_HAS_C_ATOMIC_IMP']


@public.clgen_preprocessor
def ClangPreprocess(text: str) -> str:
  return clang.Preprocess(text, CLANG_ARGS)


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
