import re

from deeplearning.clgen.preprocessors import clang


C_COMMENT_RE = re.compile(
  r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
  re.DOTALL | re.MULTILINE)


def ClangPreprocess(src: str) -> str:
  return clang.Preprocess(src, ['-x', 'c++'])


def StripComments(text: str) -> str:
  """Strip C/C++ style comments.

  written by @markus-jarderot https://stackoverflow.com/a/241506/1318051
  """

  def Replacer(match):
    s = match.group(0)
    if s.startswith('/'):
      return " "  # note: a space and not an empty string
    else:
      return s

  return C_COMMENT_RE.sub(Replacer, text)
