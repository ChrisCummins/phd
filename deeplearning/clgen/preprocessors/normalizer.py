"""Python entry point to the clgen-rewriter binary."""
import os
import subprocess
import tempfile
import typing

from absl import flags
from absl import logging

from deeplearning.clgen import errors
from lib.labm8 import bazelutil


FLAGS = flags.FLAGS

CLGEN_REWRITER = bazelutil.DataPath(
    'phd/deeplearning/clgen/native/clgen-rewriter')
assert CLGEN_REWRITER.is_file()

# On Linux we must preload the LLVM sharded libraries.
CLGEN_REWRITER_ENV = os.environ.copy()
_LIBCLANG_SO = bazelutil.DataPath(
    f'llvm_linux/lib/libclang.so', must_exist=False)
_LIBLTO_SO = bazelutil.DataPath(
    f'llvm_linux/lib/libLTO.so', must_exist=False)
if _LIBCLANG_SO.is_file() and _LIBLTO_SO.is_file():
  CLGEN_REWRITER_ENV['LD_PRELOAD'] = f'{_LIBCLANG_SO}:{_LIBLTO_SO}'

def NormalizeIdentifiers(text: str, suffix: str, cflags: typing.List[str],
                         timeout_seconds: int = 60) -> str:
  """Normalize identifiers in source code.

  An LLVM rewriter pass which renames all functions and variables with short,
  unique names. The variables and functions defined within the input text
  are rewritten, with the sequence 'A', 'B', ... 'AA', 'AB'... being used for
  function names, and the sequence 'a', 'b', ... 'aa', 'ab'... being used for
  variable names. Functions and variables which are defined in #include files
  are not renamed. Undefined function and variable names are not renamed.

  Args:
    text: The source code to rewrite.
    suffix: The suffix to append to the source code temporary file. E.g. '.c'
      for a C program.
    cflags: A list of flags to be passed to clang.
    timeout_seconds: The number of seconds to allow before killing the rewriter.

  Returns:
    Source code with identifier names normalized.

  Raises:
    RewriterException: If rewriter found nothing to rewrite.
    ClangTimeout: If rewriter fails to complete within timeout_seconds.
  """
  with tempfile.NamedTemporaryFile('w', suffix=suffix) as f:
    f.write(text)
    f.flush()
    cmd = ["timeout", "-s9", str(timeout_seconds), str(CLGEN_REWRITER),
           f.name] + ['-extra-arg=' + x for x in cflags] + ['--']
    logging.debug('$ %s', ' '.join(cmd))
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               universal_newlines=True, env=CLGEN_REWRITER_ENV)
    stdout, stderr = process.communicate()
    logging.debug(stderr)
  # If there was nothing to rewrite, rewriter exits with error code:
  EUGLY_CODE = 204
  if process.returncode == EUGLY_CODE:
    # Propagate the error:
    raise errors.RewriterException(stderr)
  elif process.returncode == 9:
    raise errors.ClangTimeout(
        f'clgen-rewriter failed to complete after {timeout_seconds}s')
  # The rewriter process can still fail because of some other compilation
  # problem, e.g. for some reason the 'enable 64bit support' pragma which should
  # be included in the shim isn't being propogated correctly to the rewriter.
  # However, the rewriter will still correctly process the input, so we ignore
  # all error codes except the one we care about (EUGLY_CODE).
  return stdout
