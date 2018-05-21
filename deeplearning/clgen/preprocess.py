#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Preprocess OpenCL files for machine learning.
"""
import contextlib
import json
import math
import os
import random
import re
import sqlite3
from io import open
from multiprocessing import Pool, cpu_count
from queue import Empty as QueueEmpty
from queue import Queue
from subprocess import PIPE, Popen, STDOUT
from tempfile import NamedTemporaryFile
from threading import Thread
from typing import Dict, List, Tuple

import progressbar
from absl import logging

from deeplearning.clgen import dbutil
from deeplearning.clgen import errors
from deeplearning.clgen import languages
from deeplearning.clgen import native
from lib.labm8 import fs


#
# Custom exceptions:
#

# Internal exceptions:


# FIXME(polyglot):
CLANG_CL_TARGETS = ['nvptx64-nvidia-nvcl', 'spir64']


# FIXME(polyglot):
def clang_cl_args(target: str = CLANG_CL_TARGETS[0], use_shim: bool = True,
                  error_limit: int = 0) -> List[str]:
  """
  Get the Clang args to compile OpenCL.

  Parameters
  ----------
  target : str
      LLVM target.
  use_shim : bool, optional
      Inject shim header.
  error_limit : int, optional
      Limit number of compiler errors.

  Returns
  -------
  List[str]
      Array of args.
  """
  # clang warnings to disable
  disabled_warnings = ['ignored-pragmas', 'implicit-function-declaration',
                       'incompatible-library-redeclaration',
                       'macro-redefined', ]

  args = ['-I' + fs.path(native.LIBCLC), '-target', target,
          '-ferror-limit={}'.format(error_limit), '-xcl'] + ['-Wno-{}'.format(x)
                                                             for x in
                                                             disabled_warnings]

  if use_shim:
    args += ['-include', native.SHIMFILE]

  return args


def strip_preprocessor_lines(src: str) -> str:
  lines = src.split('\n')

  # Strip all the includes:
  for i, line in enumerate(lines):
    if line == '# 1 "<stdin>" 2':
      break
  lines = lines[i + 1:]

  # Strip lines beginning with '#' (that's preprocessor
  # stuff):
  src = '\n'.join([line for line in lines if not line.startswith('#')])

  return src


def compiler_preprocess(src: str, compiler_args: List[str], id: str = 'anon',
                        timeout: int = 60):
  """Run input code through the compiler frontend to inline macros."""
  cmd = ["timeout", "-s9", str(timeout), native.CLANG] + compiler_args + ['-E',
                                                                          '-c',
                                                                          '-',
                                                                          '-o',
                                                                          '-']
  logging.debug('$ %s', ' '.join(cmd))
  process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
  stdout, stderr = process.communicate(src.encode('utf-8'))
  if process.returncode != 0:
    raise errors.ClangException(stderr.decode('utf-8'))

  src = stdout.decode('utf-8')

  return strip_preprocessor_lines(src).strip()


def compiler_preprocess_cl(src: str, id: str = 'anon',
                           use_shim: bool = True) -> str:
  """
  Preprocess OpenCL file.

  Inlines macros, removes comments, etc.

  Parameters
  ----------
  src : str
      OpenCL source.
  id : str, optional
      Name of OpenCL source.
  use_shim : bool, optional
      Inject shim header.

  Returns
  -------
  str
      Preprocessed source.

  Raises
  ------
  ClangException
      If compiler errors.
  """
  return compiler_preprocess(src, clang_cl_args(use_shim=use_shim), id)


def rewrite_cl(src: str, id: str = 'anon', use_shim: bool = True,
               timeout: int = 60) -> str:
  """
  Rewrite OpenCL sources.

  Renames all functions and variables with short, unique names.

  Parameters
  ----------
  src : str
      OpenCL source.
  id : str, optional
      OpenCL source name.
  use_shim : bool, optional
      Inject shim header.

  Returns
  -------
  str
      Rewritten OpenCL source.

  Raises
  ------
  RewriterException
      If rewriter fails.
  """
  # On Linux we must preload the clang library.
  env = os.environ
  if native.LIBCLANG_SO:
    env = os.environ.copy()
    env['LD_PRELOAD'] = native.LIBCLANG_SO

  # Rewriter can't read from stdin.
  with NamedTemporaryFile('w', suffix='.cl') as tmp:
    tmp.write(src)
    tmp.flush()
    cmd = (["timeout", "-s9", str(timeout), native.CLGEN_REWRITER, tmp.name] + [
      '-extra-arg=' + x for x in clang_cl_args(use_shim=use_shim)] + ['--'])
    logging.debug('$ %s', ' '.join(cmd))

    process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE,
                    universal_newlines=True, env=env)
    stdout, stderr = process.communicate()
    logging.debug(stderr)

  # If there was nothing to rewrite, rewriter exits with error code:
  EUGLY_CODE = 204
  if process.returncode == EUGLY_CODE:
    # Propagate the error:
    raise errors.RewriterException(src)
  # NOTE: the rewriter process can still fail because of some other
  # compilation problem, e.g. for some reason the 'enable 64bit
  # support' pragma which should be included in the shim isn't being
  # propogated correctly to the rewriter. However, the rewriter will
  # still correctly process the input, so we ignore all error codes
  # except the one we care about (EUGLY_CODE).
  return stdout


def compile_cl_bytecode(src: str, id: str = 'anon', use_shim: bool = True,
                        timeout: int = 60) -> str:
  """
  Compile OpenCL kernel to LLVM bytecode.

  Parameters
  ----------
  src : str
      OpenCL source.
  id : str, optional
      Name of OpenCL source.
  use_shim : bool, optional
      Inject shim header.

  Returns
  -------
  str
      Bytecode.

  Raises
  ------
  ClangException
      If compiler errors.
  """
  cmd = ["timeout", "-s9", str(timeout), native.CLANG] + clang_cl_args(
    use_shim=use_shim) + ['-emit-llvm', '-S', '-c', '-', '-o', '-']

  logging.debug('$ %s', ' '.join(cmd))
  process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE,
                  universal_newlines=True)
  stdout, stderr = process.communicate(src)

  if process.returncode != 0:
    raise errors.ClangException(stderr)
  return stdout


def gpuverify(src: str, args: list, id: str = 'anon', timeout: int = 60) -> str:
  """
  Run GPUverify over kernel.

  Parameters
  ----------
  src : str
      OpenCL source.
  id : str, optional
      OpenCL source name.

  Returns
  -------
  str
      OpenCL source.

  Raises
  ------
  GPUVerifyException
      If GPUverify finds a bug.
  InternalError
      If GPUverify fails.
  """
  # TODO(cec): Re-enable GPUVerify support.
  # from lib.labm8 import system
  # if not system.is_linux():
  #   raise errors.InternalError("GPUVerify only supported on Linux!")
  #
  # # GPUverify can't read from stdin.
  # with NamedTemporaryFile('w', suffix='.cl') as tmp:
  #   tmp.write(src)
  #   tmp.flush()
  #   cmd = ['timeout', '-s9', str(timeout), native.GPUVERIFY, tmp.name] + args
  #
  #   process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
  #   stdout, stderr = process.communicate()
  #
  # if process.returncode == -9:  # timeout signal
  #   raise errors.GPUVerifyTimeoutException(
  #     f"GPUveryify failed to complete with {timeout} seconds")
  # elif process.returncode != 0:
  #   raise errors.GPUVerifyException(stderr.decode('utf-8'))

  return src


_instcount_re = re.compile(
  r"^(?P<count>\d+) instcount - Number of (?P<type>.+)")


def parse_instcounts(txt: str) -> Dict[str, int]:
  """
  Parse LLVM opt instruction counts pass.

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


def instcounts2ratios(counts: Dict[str, int]) -> Dict[str, float]:
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


def bytecode_features(bc: str, id: str = 'anon', timeout: int = 60) -> Dict[
  str, float]:
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
  cmd = ["timeout", "-s9", str(timeout), native.OPT, '-analyze', '-stats',
         '-instcount', '-']

  # LLVM pass output pritns to stderr, so we'll pipe stderr to
  # stdout.
  logging.debug('$ %s', ' '.join(cmd))
  process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=STDOUT,
                  universal_newlines=True)
  stdout, _ = process.communicate(bc)

  if process.returncode != 0:
    raise errors.OptException(stdout)

  instcounts = parse_instcounts(stdout)
  instratios = instcounts2ratios(instcounts)

  return instratios


# Options to pass to clang-format.
#
# See: http://clang.llvm.org/docs/ClangFormatStyleOptions.html
#
clangformat_config = {'BasedOnStyle': 'Google', 'ColumnLimit': 500,
                      'IndentWidth': 2, 'AllowShortBlocksOnASingleLine': False,
                      'AllowShortCaseLabelsOnASingleLine': False,
                      'AllowShortFunctionsOnASingleLine': False,
                      'AllowShortLoopsOnASingleLine': False,
                      'AllowShortIfStatementsOnASingleLine': False,
                      'DerivePointerAlignment': False,
                      'PointerAlignment': 'Left'}


def clangformat(src: str, id: str = 'anon', timeout: int = 60) -> str:
  """
  Enforce code style on source file.

  Parameters
  ----------
  src : str
      Source code.
  id : str, optional
      Name of source file.

  Returns
  -------
  str
      Styled source.

  Raises
  ------
  ClangFormatException
      If formatting errors.
  """
  cmd = ["timeout", "-s9", str(timeout), native.CLANG_FORMAT,
         '-style={}'.format(json.dumps(clangformat_config))]
  logging.debug('$ %s', ' '.join(cmd))
  process = Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE)
  stdout, stderr = process.communicate(src.encode('utf-8'))

  if stderr:
    logging.error(stderr.decode('utf-8'))
  if process.returncode != 0:
    raise errors.ClangFormatException(stderr.decode('utf-8'))

  return stdout.decode('utf-8')


def verify_bytecode_features(bc_features: Dict[str, float],
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


def ensure_has_code(src: str) -> str:
  """
  Check that file contains actual executable code.

  Parameters
  ----------
  src : str
      OpenCL source, must be preprocessed.

  Returns
  -------
  src
      Unmodified source code.

  Raises
  ------
  NoCodeException
      If kernel is empty.
  """
  if len(src.split('\n')) < 3:
    raise errors.NoCodeException

  return src


def sanitize_prototype(src: str) -> str:
  """
  Sanitize OpenCL prototype.

  Ensures that OpenCL prototype fits on a single line.

  Parameters
  ----------
  src : str
      OpenCL source.

  Returns
  -------
  src
      Source code with sanitized prototypes.

  Returns
  -------
  str
      Sanitized OpenCL source.
  """
  # Ensure that prototype is well-formed on a single line:
  try:
    prototype_end_idx = src.index('{') + 1
    prototype = ' '.join(src[:prototype_end_idx].split())
    return prototype + src[prototype_end_idx:]
  except ValueError:
    # Ok so erm... if the '{' character isn't found, a ValueError
    # is thrown. Why would '{' not be found? Who knows, but
    # whatever, if the source file got this far through the
    # preprocessing pipeline then it's probably "good" code. It
    # could just be that an empty file slips through the cracks or
    # something.
    return src


def strip_comments(text: str):
  """
  Strip C/C++ style comments.

  written by @markus-jarderot https://stackoverflow.com/a/241506/1318051
  """

  def replacer(match):
    s = match.group(0)
    if s.startswith('/'):
      return " "  # note: a space and not an empty string
    else:
      return s

  pattern = re.compile(
    r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
    re.DOTALL | re.MULTILINE)
  return re.sub(pattern, replacer, text)


def remove_duplicate_empty_lines(text: str):
  """
  Truncate blank lines.
  """
  last_line = None
  lines = []
  for line in text.split("\n"):
    line = line.rstrip()
    if line or last_line:
      lines.append(line)
    last_line = line
  return "\n".join(lines)


def preprocess_opencl(src: str, id: str = 'anon', use_shim: bool = True,
                      use_gpuverify: bool = False) -> str:
  """
  Preprocess an OpenCL source. There are three possible outcomes:

  1. Good. Code is preprocessed and ready to be put into a training set.
  2. Bad. Code can't be preprocessed (i.e. it's "bad" OpenCL).
  3. Ugly. Code can be preprocessed but isn't useful for training
     (e.g. it's an empty file).

  Parameters
  ----------
  src : str
      The source code as a string.
  id : str, optional
      An identifying name for the source code (used in exception messages).
  use_shim : bool, optional
      Inject shim header.
  use_gpuverify : bool, optional
      Whether to run GPUVerify on the code.
  """
  # Compile to bytecode and verify features:
  logging.debug('OpenCL source = %s', src)
  bc = compile_cl_bytecode(src, id, use_shim)
  # TODO(cec): Fix llvm opt instcount invocation. There appears to be no output
  # (at least on macos homebrew version).
  #   $ /usr/local/opt/llvm/bin/clang -emit-llvm ~/foo.cc -S -o foo.ll
  #   $ /usr/local/opt/llvm/bin/opt -S -instcount -stats -analyze < foo.ll
  # bc_features = bytecode_features(bc, id)
  # verify_bytecode_features(bc_features, id)

  # Rewrite and format source:
  src = compiler_preprocess_cl(src, id, use_shim)
  src = rewrite_cl(src, id, use_shim)
  src = clangformat(src, id).strip()
  src = ensure_has_code(src)
  src = sanitize_prototype(src)

  if use_gpuverify:
    pass
    # TODO(cec): Re-enable GPUVerify.
    gpuverify(src)

  return src


def preprocess_solidity(src: str, id: str = 'anon', **kwargs) -> str:
  """
  Preprocess a solidity source.
  """
  src = strip_comments(src)
  src = remove_duplicate_empty_lines(src)
  src = clangformat(src)
  return src


def preprocess_glsl(src: str, id: str = 'anon', **kwargs) -> str:
  """
  Process a GLSL source.
  """
  src = compiler_preprocess(src, [], id)
  src = remove_duplicate_empty_lines(src)
  src = clangformat(src)
  return src


def preprocess(src: str, id: str = "anon",
               lang: languages.Language = languages.Language.OPENCL,
               **lang_opts) -> str:
  """
  Preprocess a file. There are three possible outcomes:

  1. Good. Code is preprocessed and ready to be put into a training set.
  2. Bad. Code can't be preprocessed (i.e. it's "bad" code).
  3. Ugly. Code can be preprocessed but isn't useful for training
     (e.g. it's an empty file).

  Parameters
  ----------
  src : str
      The source code as a string.
  id : str, optional
      An identifying name for the source code (used in exception messages).

  Returns
  -------
  str
      Preprocessed source code as a string.

  Raises
  ------
  BadCodeException
      If code is bad (see above).
  UglyCodeException
      If code is ugly (see above).
  errors.InternalException
      In case of some other error.
  """
  if lang == languages.Language.OPENCL:
    return preprocess_opencl(src, id, **lang_opts)
  elif lang == languages.Language.SOLIDITY:
    return preprocess_solidity(src, id, **lang_opts)
  elif lang == languages.Language.GLSL:
    return preprocess_glsl(src, id, **lang_opts)
  else:
    raise errors.ValueError(f"unsuporrted language '{lang}'")


def preprocess_for_db(src: str, **preprocess_opts) -> Tuple[int, str]:
  """
  Preprocess source code for import into contentdb.

  Parameters
  ----------
  src : str
      Source to preprocess.
  **preprocess_opts
      Preprocessing options.

  Returns
  -------
  Tuple[int, str]
      The status of the preprocessed code, and the preprocess output.
  """
  try:
    # Try and preprocess it:
    status = 0
    contents = preprocess(src, **preprocess_opts)
  except BadCodeException as e:
    status = 1
    contents = str(e)
  except UglyCodeException as e:
    status = 2
    contents = str(e)

  return status, contents


def preprocess_file(path: str, inplace: bool = False,
                    **preprocess_opts) -> None:
  """
  Preprocess a file.

  Prints output to stdout by default. If preprocessing fails, this function
  exits.

  Parameters
  ----------
  path : str
      String path to file.
  inplace : bool, optional
      If True, overwrite input file.
  """
  with open(path) as infile:
    contents = infile.read()
  try:
    out = preprocess(contents, **preprocess_opts)
    if inplace:
      with open(path, 'w') as outfile:
        outfile.write(out)
    else:
      print(out)
  except BadCodeException as e:
    logging.fatal(e, ret=1)
  except UglyCodeException as e:
    logging.fatal(e, ret=2)


def _preprocess_inplace_worker(path: str) -> None:
  """worker function for preprocess_inplace()"""
  logging.info('preprocess', path)
  preprocess_file(path, inplace=True)


@contextlib.contextmanager
def terminating(thing):
  """
  Context manager to terminate object at end of scope.
  """
  try:
    yield thing
  finally:
    thing.terminate()


def preprocess_inplace(paths: List[str], max_num_workers: int = cpu_count(),
                       max_attempts: int = 100, attempt: int = 1) -> None:
  """
  Preprocess a list of files in place.

  Parameters
  ----------
  paths : List[str]
      List of paths.
  max_num_workers : int, optional
      Number of processes to spawn.
  max_attempts : int, optional
      In case of an OSError or TimeoutError, this number of attempts will be
      made.
  """
  if attempt > max_attempts:
    raise errors.InternalError(
      f"Failed to process files after {max_attempts} attempts")
  elif attempt > 1:
    logging.warning("preprocess attempt #.", attempt)

  num_workers = min(len(paths), max_num_workers)

  try:
    logging.info('spawned', num_workers, 'worker threads to process',
                 len(paths), 'files ...')
    with terminating(Pool(num_workers)) as pool:
      pool.map(_preprocess_inplace_worker, paths)
  except (OSError, TimeoutError) as e:
    logging.error(e)

    # Try again with fewer threads.
    # See: https://github.com/ChrisCummins/clgen/issues/64
    max_num_workers = max(int(max_num_workers / 2), 1)
    preprocess_inplace(paths, max_num_workers=max_num_workers,
                       attempt=attempt + 1, max_attempts=max_attempts)


class PreprocessWorker(Thread):
  """ preprocessor worker thread """

  def __init__(self, jobs: List[Dict], queue: Queue):
    super(PreprocessWorker, self).__init__()
    self.jobs = jobs
    self.queue = queue

  def run(self):
    while self.jobs:
      job = self.jobs.pop(0)

      kid = job["id"]
      src = job["src"]
      preprocess_opts = job["preprocess_opts"]

      status, contents = preprocess_for_db(src, id=id, **preprocess_opts)

      result = {"id": kid, "status": status, "contents": contents}

      self.queue.put(result)


def _preprocess_db(db_path: str, max_num_workers: int = cpu_count(),
                   max_attempts: int = 100, attempt: int = 1,
                   **preprocess_opts) -> None:
  """
  Preprocess OpenCL dataset.

  Parameters
  ----------
  db_path : str
      OpenCL kernels dataset.
  max_num_workers : int, optional
      Number of processes to spawn.
  max_attempts : int, optional
      In case of an OSError or TimeoutError, this number of attempts will be
      made.
  """
  if attempt > max_attempts:
    raise errors.InternalError(
      f"failed to preprocess files after {max_attempts} attempts")

  logging.debug("determining jobs")

  contentfiles = set(dbutil.kernel_ids(db_path, "ContentFiles"))
  preprocessedfiles = set(dbutil.kernel_ids(db_path, "PreprocessedFiles"))

  ncontentfiles = len(contentfiles)
  npreprocessedfiles = len(preprocessedfiles)

  todo = contentfiles - preprocessedfiles
  ntodo = len(todo)

  # check we have something to do
  if not ntodo:
    return

  todo_ratio = ntodo / ncontentfiles

  logging.info(
    "{ntodo} ({todo_ratio:.1%}) samples need preprocessing".format(**vars()))

  logging.debug("creating jobs")

  # Determine if we need to inline kernels when creating jobs
  db = sqlite3.connect(db_path)
  c = db.cursor()
  c.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name='ContentMeta';")
  meta_table = c.fetchone()
  c.close()
  db.close()
  if meta_table:
    get_kernel = lambda kid: dbutil.get_inlined_kernel(db_path, kid,
                                                       lang=preprocess_opts[
                                                         "lang"])
  else:
    get_kernel = lambda kid: dbutil.get_kernel(db_path, kid,
                                               table="ContentFiles")

  # create jobs
  jobs = [
    {"id": kid, "src": get_kernel(kid), "preprocess_opts": preprocess_opts, }
    for kid in todo]

  random.shuffle(jobs)

  # split size
  worker_njobs = math.ceil(ntodo / max_num_workers)

  # producer-consumer queue
  queue = Queue(maxsize=128)

  logging.debug(f"assigning {ntodo} jobs to {max_num_workers} threads")

  try:
    # our worker threads. these busy little bees will do the heavy lifting
    # of preprocessing the contentfiles, pushing their results onto
    # the queue
    producers = [PreprocessWorker(jobs[i:i + worker_njobs], queue) for i in
                 range(0, ntodo, worker_njobs)]

    # fly, my pretties, fly!
    for producer in producers:
      producer.start()

    # consume the results from the worker threads from the main thread
    for i in progressbar.ProgressBar()(range(ntodo)):
      # pull a fresh result from the queue (block if necessary)
      try:
        result = queue.get(timeout=90)
      except QueueEmpty as e:
        raise errors.TimeoutError('failed to fetch result after 90 seconds. '
                                  'something went wrong') from e

      # insert result into database
      db = dbutil.connect(db_path)
      c = db.cursor()
      c.execute("INSERT INTO PreprocessedFiles VALUES(?,?,?)",
                (result["id"], result["status"], result["contents"]))
      c.close()
      db.commit()
      db.close()

    for producer in producers:
      producer.join()

  except (OSError, TimeoutError) as e:
    logging.error(e)

    if attempt > 2 and not i:
      logging.warning("no progress has been made since previous attempt. "
                      "I'm not going to try another attempt.")
      return

    # Try again with fewer threads.
    # See: https://github.com/ChrisCummins/clgen/issues/64
    max_num_workers = max(int(max_num_workers / 2), 1)
    _preprocess_db(db_path, max_num_workers=max_num_workers,
                   attempt=attempt + 1, max_attempts=max_attempts,
                   **preprocess_opts)


def preprocess_db(db_path: str, **preprocess_opts) -> bool:
  """
  Preprocess database contents.

  Parameters
  ----------
  db_path : str
      Path to database.

  Returns
  -------
  bool
      True if modified, false if no work needed.
  """
  db = dbutil.connect(db_path)

  modified = dbutil.is_modified(db)
  if modified:
    _preprocess_db(db_path, **preprocess_opts)
    dbutil.set_modified_status(db, modified)
    return True
  else:
    return False
