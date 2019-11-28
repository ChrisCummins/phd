# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""This file contains utility code for working with clang.

This module does not expose any preprocessor functions for CLgen. It contains
wrappers around Clang binaries, which preprocessor functions can use to
implement specific behavior. See deeplearning.clgen.preprocessors.cxx.Compile()
for an example.
"""
import json
import re
import subprocess
import sys
import tempfile
import typing

from compilers.llvm import clang_format
from compilers.llvm import llvm
from deeplearning.clgen import errors
from labm8.py import app
from labm8.py import bazelutil

FLAGS = app.FLAGS

_LLVM_REPO = 'llvm_mac' if sys.platform == 'darwin' else 'llvm_linux'
# Path to clang binary.
CLANG = bazelutil.DataPath(f'{_LLVM_REPO}/bin/clang')
# The marker used to mark stdin from clang pre-processor output.
CLANG_STDIN_MARKER = re.compile(r'# \d+ "<stdin>" 2')
# Options to pass to clang-format.
# See: http://clang.llvm.org/docs/ClangFormatStyleOptions.html
CLANG_FORMAT_CONFIG = {
    'BasedOnStyle': 'Google',
    'ColumnLimit': 5000,
    'IndentWidth': 2,
    'AllowShortBlocksOnASingleLine': False,
    'AllowShortCaseLabelsOnASingleLine': False,
    'AllowShortFunctionsOnASingleLine': False,
    'AllowShortLoopsOnASingleLine': False,
    'AllowShortIfStatementsOnASingleLine': False,
    'DerivePointerAlignment': False,
    'PointerAlignment': 'Left',
    'BreakAfterJavaFieldAnnotations': True,
    'BreakBeforeInheritanceComma': False,
    'BreakBeforeTernaryOperators': False,
    'AlwaysBreakAfterReturnType': 'None',
    'AlwaysBreakAfterDefinitionReturnType': 'None',
}


def StripPreprocessorLines(src: str) -> str:
  """Strip preprocessor remnants from clang frontend output.

  Args:
    src: Clang frontend output.

  Returns:
    The output with preprocessor output stripped.
  """
  lines = src.split('\n')
  # Determine when the final included file ends.
  for i in range(len(lines) - 1, -1, -1):
    if CLANG_STDIN_MARKER.match(lines[i]):
      break
  else:
    return ''
  # Strip lines beginning with '#' (that's preprocessor stuff):
  return '\n'.join([line for line in lines[i:] if not line.startswith('#')])


def Preprocess(src: str,
               cflags: typing.List[str],
               timeout_seconds: int = 60,
               strip_preprocessor_lines: bool = True):
  """Run input code through the compiler frontend to inline macros.

  This uses the repository clang binary.

  Args:
    src: The source code to preprocess.
    cflags: A list of flags to be passed to clang.
    timeout_seconds: The number of seconds to allow before killing clang.
    strip_preprocessor_lines: Whether to strip the extra lines introduced by
      the preprocessor.

  Returns:
    The preprocessed code.

  Raises:
    ClangException: In case of an error.
    ClangTimeout: If clang does not complete before timeout_seconds.
  """
  cmd = [
      'timeout', '-s9',
      str(timeout_seconds),
      str(CLANG), '-E', '-c', '-', '-o', '-'
  ] + cflags
  app.Log(2, '$ %s', ' '.join(cmd))
  process = subprocess.Popen(cmd,
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
  stdout, stderr = process.communicate(src)
  if process.returncode == 9:
    raise errors.ClangTimeout(
        f'Clang preprocessor timed out after {timeout_seconds}s')
  elif process.returncode != 0:
    raise errors.ClangException(stderr)
  if strip_preprocessor_lines:
    return StripPreprocessorLines(stdout)
  else:
    return stdout


def CompileLlvmBytecode(src: str,
                        suffix: str,
                        cflags: typing.List[str],
                        timeout_seconds: int = 60) -> str:
  """Compile input code into textual LLVM byte code.

  Args:
    src: The source code to compile.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    cflags: A list of flags to be passed to clang.
    timeout_seconds: The number of seconds to allow before killing clang.

  Returns:
    The textual LLVM byte code.

  Raises:
    ClangException: In case of an error.
    ClangTimeout: If clang does not complete before timeout_seconds.
  """
  builtin_cflags = ['-S', '-emit-llvm', '-o', '-']
  with tempfile.NamedTemporaryFile(
      'w', prefix='phd_deeplearning_clgen_preprocessors_clang_',
      suffix=suffix) as f:
    f.write(src)
    f.flush()
    cmd = ['timeout', '-s9',
           str(timeout_seconds),
           str(CLANG), f.name] + builtin_cflags + cflags
    app.Log(2, '$ %s', ' '.join(cmd))
    process = subprocess.Popen(cmd,
                               stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
    stdout, stderr = process.communicate()
  if process.returncode == 9:
    raise errors.ClangTimeout(f'Clang timed out after {timeout_seconds}s')
  elif process.returncode != 0:
    raise errors.ClangException(stderr)
  return stdout


def ClangFormat(text: str, suffix: str, timeout_seconds: int = 60) -> str:
  """Run clang-format on a source to enforce code style.

  Args:
    text: The source code to run through clang-format.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    timeout_seconds: The number of seconds to allow clang-format to run for.

  Returns:
    The output of clang-format.

  Raises:
    ClangFormatException: In case of an error.
    ClangTimeout: If clang-format does not complete before timeout_seconds.
  """
  try:
    return clang_format.Exec(
        text, suffix, ['-style={}'.format(json.dumps(CLANG_FORMAT_CONFIG))],
        timeout_seconds)
  except llvm.LlvmTimeout:
    raise errors.ClangTimeout(
        f'Clang-format timed out after {timeout_seconds}s')
  except clang_format.ClangFormatException as e:
    raise errors.ClangFormatException(str(e))
