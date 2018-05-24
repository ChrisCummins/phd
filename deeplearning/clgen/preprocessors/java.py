"""Preprocessor modules for the Java programming language."""
from absl import flags

from deeplearning.clgen.preprocessors import clang
from deeplearning.clgen.preprocessors import preprocessors


FLAGS = flags.FLAGS


@preprocessors.clgen_preprocessor
def ClangFormat(text: str) -> str:
  return clang.ClangFormat(text, '.java')
