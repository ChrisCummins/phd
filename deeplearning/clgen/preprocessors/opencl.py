"""Preprocessor passes for the OpenCL programming language."""
import os
import subprocess
import tempfile
import typing

from absl import logging

from deeplearning.clgen import errors
from deeplearning.clgen import native
from deeplearning.clgen.preprocessors import clang


def GetClangArgs(use_shim: bool = False, error_limit: int = 0) -> typing.List[
  str]:
  """Get the arguments to pass to clang for handling OpenCL.

  Args:
    use_shim: If true, inject the shim OpenCL header.
    error_limit: The number of errors to print before arboting

  Returns:
    A list of command line arguments to pass to Popen().
  """
  # Clang warnings to disable.
  disabled_warnings = ['ignored-pragmas', 'implicit-function-declaration',
                       'incompatible-library-redeclaration',
                       'macro-redefined', ]
  args = ['-I' + str(native.LIBCLC), '-target', 'nvptx64-nvidia-nvcl',
          f'-ferror-limit={error_limit}', '-xcl'] + ['-Wno-{}'.format(x) for x
                                                     in disabled_warnings]
  if use_shim:
    args += ['-include', str(native.SHIMFILE)]
  return args


def ClangPreprocess(src: str, use_shim: bool = True) -> str:
  """Preprocess OpenCL source.

  Inline macros, removes comments, etc.

  Args:
    src: OpenCL source.
    id: Name of OpenCL source.
    use_shim: Inject shim header.

  Returns:
    Preprocessed source.
  """
  return clang.Preprocess(src, GetClangArgs(use_shim=use_shim))


def NormalizeIdentifiers(src: str, timeout_seconds: int = 60,
                         use_shim: bool = True) -> str:
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
  with tempfile.NamedTemporaryFile('w', suffix='.cl') as tmp:
    tmp.write(src)
    tmp.flush()
    cmd = (["timeout", "-s9", str(timeout_seconds), native.CLGEN_REWRITER,
            tmp.name] + ['-extra-arg=' + x for x in
                         GetClangArgs(use_shim=use_shim)] + ['--'])
    logging.debug('$ %s', ' '.join(cmd))

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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


def GpuVerify(src: str, args: list, id: str = 'anon', timeout: int = 60) -> str:
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
  # with tempfile.NamedTemporaryFile('w', suffix='.cl') as tmp:
  #   tmp.write(src)
  #   tmp.flush()
  #   cmd = ['timeout', '-s9', str(timeout), native.GPUVERIFY, tmp.name] + args
  #
  #   process = subprocess.Popen(cmd, stdin=subprocess.PIPE,
  # stdout=subprocess.PIPE, stderr=subprocess.PIPE)
  #   stdout, stderr = process.communicate()
  #
  # if process.returncode == -9:  # timeout signal
  #   raise errors.GPUVerifyTimeoutException(
  #     f"GPUveryify failed to complete with {timeout} seconds")
  # elif process.returncode != 0:
  #   raise errors.GPUVerifyException(stderr.decode('utf-8'))

  return src


def SanitizeKernelPrototype(src: str) -> str:
  """Sanitize OpenCL prototype.

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
