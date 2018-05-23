"""This file contains utility code for working with clang."""
import json
import pathlib
import re
import subprocess
import typing

from absl import flags
from absl import logging

from config import getconfig
from deeplearning.clgen import dbutil
from deeplearning.clgen import errors


FLAGS = flags.FLAGS

_config = getconfig.GetGlobalConfig()
# Path to clang binary.
CLANG = _config.paths.cc
# Path to clang-format binary.
CLANG_FORMAT = _config.paths.clang_format
# Path to opt binary.
OPT = _config.paths.opt
# The marker used to mark stdin from clang pre-processor output.
CLANG_STDIN_MARKER = re.compile(r'# \d+ "<stdin>" 2')
# Options to pass to clang-format.
# See: http://clang.llvm.org/docs/ClangFormatStyleOptions.html
CLANG_FORMAT_CONFIG = {'BasedOnStyle': 'Google', 'ColumnLimit': 500,
                       'IndentWidth': 2, 'AllowShortBlocksOnASingleLine': False,
                       'AllowShortCaseLabelsOnASingleLine': False,
                       'AllowShortFunctionsOnASingleLine': False,
                       'AllowShortLoopsOnASingleLine': False,
                       'AllowShortIfStatementsOnASingleLine': False,
                       'DerivePointerAlignment': False,
                       'PointerAlignment': 'Left'}


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
    if CLANG_STDIN_MARKER.match(lines[i]):
      break
  else:
    return ''
  # Strip lines beginning with '#' (that's preprocessor stuff):
  return '\n'.join([line for line in lines[i:] if not line.startswith('#')])


def Preprocess(src: str, cflags: typing.List[str], timeout_seconds: int = 60,
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


def CompileLlvmBytecode(src: str, cflags: typing.List[str],
                        timeout_seconds: int = 60) -> str:
  """Compile input code into textual LLVM byte code.

  Args:
    src: The source code to compiler.
    cflags: A list of flags to be passed to clang.
    timeout_seconds: The number of seconds to allow before killing clang.

  Returns:
    The textual LLVM byte code.

  Raises:
    ClangException: In case of an error.
    ClangTimeout: If clang does not complete before timeout_seconds.
  """
  # TODO(cec): I haven't the willpower to get this work again.
  builtin_cflags = ['-S', '-emit-llvm', '-o', '-']
  path = pathlib.Path('/tmp/input')
  try:
    with open(path, 'w') as f:
      f.write(src)
    with open(path) as f:
      print(f.read())
    cmd = ['timeout', '-s9', str(timeout_seconds),
           CLANG, ] + builtin_cflags + cflags + [str(path)]
    logging.debug('$ %s', ' '.join(cmd))
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)
    stdout, stderr = process.communicate()
  finally:
    path.unlink()
  if process.returncode == 9:
    raise errors.ClangTimeout(f'Clang timed out after {timeout_seconds}s')
  elif process.returncode != 0:
    raise errors.ClangException(stderr)
  return stdout


def ClangFormat(src: str, timeout_seconds: int = 60) -> str:
  """Run clang-format on a source to enforce code style.

  Args:
    src: The source code to run through clang-format.
    timeout_seconds: The number of seconds to allow clang-format to run for.

  Returns:
    The output of clang-format.

  Raises:
    ClangFormatException: In case of an error.
    ClangTimeout: If clang-format does not complete before timeout_seconds.
  """
  cmd = ["timeout", "-s9", str(timeout_seconds), CLANG_FORMAT,
         '-style={}'.format(json.dumps(CLANG_FORMAT_CONFIG))]
  logging.debug('$ %s', ' '.join(cmd))
  process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, universal_newlines=True)
  stdout, stderr = process.communicate(src)
  if process.returncode == 9:
    raise errors.ClangTimeout(
      f'Clang-format timed out after {timeout_seconds}s')
  elif process.returncode != 0:
    raise errors.ClangFormatException(stderr)
  return stdout


_instcount_re = re.compile(
  r"^(?P<count>\d+) instcount - Number of (?P<type>.+)")


def parse_instcounts(txt: str) -> typing.Dict[str, int]:
  """Parse LLVM opt instruction counts pass.

  Parameters
  ----------
  txt : str
      LLVM output.

  Returns
  -------
  Dict[str, int]
      key, value pairs, where key is instruction type and value is
      instruction type count.
  """
  lines = [x.strip() for x in txt.split("\n")]
  counts = {}

  # Build a list of counts for each type.
  for line in lines:
    match = re.search(_instcount_re, line)
    if match:
      count = int(match.group("count"))
      key = match.group("type")
      if key in counts:
        counts[key].append(count)
      else:
        counts[key] = [count]

  # Sum all counts.
  for key in counts:
    counts[key] = sum(counts[key])

  return counts


def instcounts2ratios(counts: typing.Dict[str, int]) -> typing.Dict[str, float]:
  """
  Convert instruction counts to instruction densities.

  If, for example, there are 10 instructions and 2 addition instructions,
  then the instruction density of add operations is .2.

  Parameters
  ----------
  counts : Dict[str, int]
      Key value pairs of instruction types and counts.

  Returns
  -------
  ratios : Dict[str, float]
      Key value pairs of instruction types and densities.
  """
  if not len(counts):
    return {}

  ratios = {}
  total_key = "instructions (of all types)"
  non_ratio_keys = [total_key]
  total = float(counts[total_key])

  for key in non_ratio_keys:
    ratios[dbutil.escape_sql_key(key)] = counts[key]

  for key in counts:
    if key not in non_ratio_keys:
      # Copy count
      ratios[dbutil.escape_sql_key(key)] = counts[key]
      # Insert ratio
      ratio = float(counts[key]) / total
      ratios[dbutil.escape_sql_key('ratio_' + key)] = ratio

  return ratios


def bytecode_features(bc: str, id: str = 'anon', timeout: int = 60) -> \
    typing.Dict[str, float]:
  """
  Extract features from bytecode.

  Parameters
  ----------
  bc : str
      LLVM bytecode.
  id : str, optional
      Name of OpenCL source.

  Returns
  -------
  Dict[str, float]
      Key value pairs of instruction types and densities.

  Raises
  ------
  OptException
      If LLVM opt pass errors.
  """
  cmd = ["timeout", "-s9", str(timeout), OPT, '-analyze', '-stats',
         '-instcount', '-']

  # LLVM pass output pritns to stderr, so we'll pipe stderr to
  # stdout.
  logging.debug('$ %s', ' '.join(cmd))
  process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, universal_newlines=True)
  stdout, _ = process.communicate(bc)

  if process.returncode != 0:
    raise errors.OptException(stdout)

  instcounts = parse_instcounts(stdout)
  instratios = instcounts2ratios(instcounts)

  return instratios


def verify_bytecode_features(bc_features: typing.Dict[str, float],
                             id: str = 'anon') -> None:
  """
  Verify LLVM bytecode features.

  Parameters
  ----------
  bc_features : Dict[str, float]
      Bytecode features.
  id : str, optional
      Name of OpenCL source.

  Raises
  ------
  InstructionCountException
      If verification errors.
  """
  # The minimum number of instructions before a kernel is discarded
  # as ugly.
  min_num_instructions = 1
  num_instructions = bc_features.get('instructions_of_all_types', 0)

  if num_instructions < min_num_instructions:
    raise errors.InstructionCountException(
      'Code contains {} instructions. The minimum allowed is {}'.format(
        num_instructions, min_num_instructions))
