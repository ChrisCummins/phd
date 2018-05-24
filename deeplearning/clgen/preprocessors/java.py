"""Preprocessor modules for the Java programming language."""
from absl import flags

from deeplearning.clgen.preprocessors import clang
from deeplearning.clgen.preprocessors import public


FLAGS = flags.FLAGS


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
