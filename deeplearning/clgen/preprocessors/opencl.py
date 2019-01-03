"""Preprocessor passes for the OpenCL programming language."""
import typing

from absl import flags

from deeplearning.clgen.preprocessors import clang
from deeplearning.clgen.preprocessors import normalizer
from deeplearning.clgen.preprocessors import public
from labm8 import bazelutil


FLAGS = flags.FLAGS

LIBCLC = bazelutil.DataPath('phd/third_party/libclc/generic/include')
OPENCL_H = bazelutil.DataPath('phd/deeplearning/clgen/data/include/opencl.h')
SHIMFILE = bazelutil.DataPath(
    'phd/deeplearning/clgen/data/include/opencl-shim.h')


def GetClangArgs(use_shim: bool) -> typing.List[str]:
  """Get the arguments to pass to clang for handling OpenCL.

  Args:
    use_shim: If true, inject the shim OpenCL header.
    error_limit: The number of errors to print before arboting

  Returns:
    A list of command line arguments to pass to Popen().
  """
  args = ['-I' + str(LIBCLC), '-include', str(OPENCL_H),
          '-target', 'nvptx64-nvidia-nvcl', f'-ferror-limit=1', '-xcl',
          '-Wno-ignored-pragmas', '-Wno-implicit-function-declaration',
          '-Wno-incompatible-library-redeclaration', '-Wno-macro-redefined',
          '-Wno-unused-parameter']
  if use_shim:
    args += ['-include', str(SHIMFILE)]
  return args


def _ClangPreprocess(text: str, use_shim: bool) -> str:
  """Private preprocess OpenCL source implementation.

  Inline macros, removes comments, etc.

  Args:
    text: OpenCL source.
    use_shim: Inject shim header.

  Returns:
    Preprocessed source.
  """
  return clang.Preprocess(text, GetClangArgs(use_shim=use_shim))


@public.clgen_preprocessor
def ClangPreprocess(text: str) -> str:
  """Preprocessor OpenCL source.

  Args:
    text: OpenCL source to preprocess.

  Returns:
    Preprocessed source.
  """
  return _ClangPreprocess(text, False)


@public.clgen_preprocessor
def ClangPreprocessWithShim(text: str) -> str:
  """Preprocessor OpenCL source with OpenCL shim header injection.

  Args:
    text: OpenCL source to preprocess.

  Returns:
    Preprocessed source.
  """
  return _ClangPreprocess(text, True)


@public.clgen_preprocessor
def Compile(text: str) -> str:
  """Check that the OpenCL source compiles.

  This does not modify the input.

  Args:
    text: OpenCL source to check.

  Returns:
    Unmodified OpenCL source.
  """
  # We must override the flag -Wno-implicit-function-declaration from
  # GetClangArgs() to ensure that undefined functions are treated as errors.
  clang.CompileLlvmBytecode(text, '.cl', GetClangArgs(use_shim=False) + [
    '-Werror=implicit-function-declaration'])
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
  return clang.ClangFormat(text, '.cl')


@public.clgen_preprocessor
def NormalizeIdentifiers(text: str) -> str:
  """Normalize identifiers in OpenCL source code.

  Args:
    text: The source code to rewrite.

  Returns:
    Source code with identifier names normalized.

  Raises:
    RewriterException: If rewriter found nothing to rewrite.
    ClangTimeout: If rewriter fails to complete within timeout_seconds.
  """
  return normalizer.NormalizeIdentifiers(
      text, '.cl', GetClangArgs(use_shim=False))


# TODO(cec): Re-enable GPUVerify support.
# def GpuVerify(src: str, args: list, id: str = 'anon', timeout: int = 60) ->
#  str:
#   """
#   Run GPUverify over kernel.
#
#   Parameters
#   ----------
#   src : str
#       OpenCL source.
#   id : str, optional
#       OpenCL source name.
#
#   Returns
#   -------
#   str
#       OpenCL source.
#
#   Raises
#   ------
#   GPUVerifyException
#       If GPUverify finds a bug.
#   InternalError
#       If GPUverify fails.
#   """
#   from labm8 import system
#   if not system.is_linux():
#     raise errors.InternalError("GPUVerify only supported on Linux!")
#
#   # GPUverify can't read from stdin.
#   with tempfile.NamedTemporaryFile('w', suffix='.cl') as tmp:
#     tmp.write(src)
#     tmp.flush()
#     cmd = ['timeout', '-s9', str(timeout), GPUVERIFY, tmp.name] + args
#
#     process = subprocess.Popen(cmd, stdin=subprocess.PIPE,
#   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     stdout, stderr = process.communicate()
#
#   if process.returncode == -9:  # timeout signal
#     raise errors.GPUVerifyTimeoutException(
#       f"GPUveryify failed to complete with {timeout} seconds")
#   elif process.returncode != 0:
#     raise errors.GPUVerifyException(stderr.decode('utf-8'))
#   return src


@public.clgen_preprocessor
def SanitizeKernelPrototype(text: str) -> str:
  """Sanitize OpenCL prototype.

  Ensures that OpenCL prototype fits on a single line.

  Args:
    text: OpenCL source.

  Returns:
    Source code with sanitized prototypes.
  """
  # Ensure that prototype is well-formed on a single line:
  try:
    prototype_end_idx = text.index('{') + 1
    prototype = ' '.join(text[:prototype_end_idx].split())
    return prototype + text[prototype_end_idx:]
  except ValueError:
    # Ok so erm... if the '{' character isn't found, a ValueError
    # is thrown. Why would '{' not be found? Who knows, but
    # whatever, if the source file got this far through the
    # preprocessing pipeline then it's probably "good" code. It
    # could just be that an empty file slips through the cracks or
    # something.
    return text


@public.clgen_preprocessor
def StripDoubleUnderscorePrefixes(text: str) -> str:
  """Remove the optional __ qualifiers on OpenCL keywords.

  The OpenCL spec allows __ prefix for OpenCL keywords, e.g. '__global' and
  'global' are equivalent. This preprocessor removes the '__' prefix on those
  keywords.

  Args:
    text: The OpenCL source to preprocess.

  Returns:
    OpenCL source with __ stripped from OpenCL keywords.
  """
  # List of keywords taken from the OpenCL 1.2. specification, page 169.
  replacements = {
    '__const': 'const',
    '__constant': 'constant',
    '__global': 'global',
    '__kernel': 'kernel',
    '__local': 'local',
    '__private': 'private',
    '__read_only': 'read_only',
    '__read_write': 'read_write',
    '__write_only': 'write_only',
  }
  for old, new in replacements.items():
    text = text.replace(old, new)
  return text
