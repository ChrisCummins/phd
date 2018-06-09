"""Preprocessor modules for the Java programming language."""

import pathlib
import re
import subprocess
import tempfile
import typing

from absl import flags
from absl import logging

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import clang
from deeplearning.clgen.preprocessors import public


FLAGS = flags.FLAGS

CLASS_NAME_RE = re.compile(r'public\s+class\s+(\w+)')


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
  return clang.ClangFormat(text, '.java')


def Javac(text: str, class_name: str, cflags: typing.List[str],
          timeout_seconds: int = 60) -> str:
  """Run code through javac.

  Args:
    text: The code to compile.
    class_name: The name of the class defined in the file.
    cflags: Additional options passed to javac.
    timeout_seconds: The number of seconds to wait before killing javac.

  Returns:
    The unmodified input code.
  """
  with tempfile.TemporaryDirectory('w', prefix='clgen_javac_') as d:
    path = pathlib.Path(d) / (class_name + '.java')
    with open(path, 'w') as f:
      f.write(text)
    cmd = ['timeout', '-s9', str(timeout_seconds), 'javac', f.name] + cflags
    logging.debug('$ %s', ' '.join(cmd))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = process.communicate()
  if process.returncode == 9:
    raise errors.BadCodeException(f'Javac timed out after {timeout_seconds}s')
  elif process.returncode != 0:
    raise errors.BadCodeException(stderr)
  return text


@public.clgen_preprocessor
def Compile(text: str) -> str:
  """A preprocessor which attempts to compile a class file for the given code.

  Args:
    text: Code to compile.

  Returns:
    The input code, unmodified.
  """
  match = CLASS_NAME_RE.search(text)
  if not match:
    raise errors.BadCodeException('Failed to determine class name')
  class_name = match.group(1)
  return Javac(text, class_name, [])
