"""This file contains utility code for working with clang."""
import subprocess
from typing import List

from absl import flags
from absl import logging

from config import getconfig
from deeplearning.clgen import errors


FLAGS = flags.FLAGS

_config = getconfig.GetGlobalConfig()
CLANG = _config.paths.cc
CLANG_FORMAT = _config.paths.clang_format


def StripPreprocessorLines(src: str) -> str:
  """Strip preprocessor remnants from clang frontend output.

  Args:
    src: Clang frontend output.

  Returns:
    The output with preprocessor output stripped.
  """
  lines = src.split('\n')
  i = 0
  # Determine when the final included file ends.
  for i in range(len(lines) - 1, -1, -1):
    if lines[i] == '# 1 "<stdin>" 2':
      break
  else:
    return ''
  # Strip lines beginning with '#' (that's preprocessor stuff):
  return '\n'.join([line for line in lines[i:] if not line.startswith('#')])


def Preprocess(src: str, cflags: List[str], timeout_seconds: int = 60,
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

  """
  builtin_cflags = ['-E', '-c', '-', '-o', '-']
  cmd = ['timeout', '-s9', str(timeout_seconds),
         CLANG] + builtin_cflags + cflags
  logging.debug('$ %s', ' '.join(cmd))
  process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, universal_newlines=True)
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
