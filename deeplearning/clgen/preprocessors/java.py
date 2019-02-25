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
from labm8 import bazelutil

FLAGS = flags.FLAGS

CLASS_NAME_RE = re.compile(r'public\s+class\s+(\w+)')

# Path to the compiled java rewriter.
JAVA_REWRITER = bazelutil.DataPath(
    'phd/deeplearning/clgen/preprocessors/JavaRewriter')


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


def Javac(text: str,
          class_name: str,
          cflags: typing.List[str],
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
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True)
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


@public.clgen_preprocessor
def WrapMethodInClass(text: str) -> str:
  """A preprocessor which wraps a method into a class definition.

  Args:
    text: Method to wrap in a class.

  Returns:
    A class definition.
  """
  return f"""\
public class A {{
  {text}
}}
"""


@public.clgen_preprocessor
def InsertShimImports(text: str) -> str:
  """A preprocessor which inserts a set of shim imports.

  Args:
    text: Text to prepend imports to.

  Returns:
    The text with imports prepended.
  """
  return f"""\
import java.io.*;
import java.nio.charset.*;
import java.nio.file.*;
import java.util.*;
import java.time.format.*;

{text}  
"""


@public.clgen_preprocessor
def JavaRewrite(text: str) -> str:
  """Run the Java rewriter on the text.

  Args:
    text: The source code to rewrite.

  Returns:
    Source code with identifier names normalized.

  Raises:
    RewriterError: If rewriter found nothing to rewrite.
    ClangTimeout: If rewriter fails to complete within timeout_seconds.
  """
  cmd = ['timeout', '-s9', '60', str(JAVA_REWRITER)]
  process = subprocess.Popen(
      cmd,
      stdin=subprocess.PIPE,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      universal_newlines=True)
  logging.debug('$ %s', ' '.join(cmd))
  stdout, stderr = process.communicate(text)
  if process.returncode == 9:
    raise errors.RewriterException('JavaRewriter failed to complete after 60s')
  elif process.returncode:
    raise errors.RewriterException(stderr)
  return stdout.strip() + '\n'
