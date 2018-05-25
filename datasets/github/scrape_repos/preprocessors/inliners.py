"""Preprocessors to inline includes."""
import pathlib
import typing

from absl import flags


FLAGS = flags.FLAGS


def InlineHeaders(path: pathlib.Path, import_root: pathlib.Path,
                  candidate_includes: typing.Set[pathlib.Path],
                  include_regex: typing.re, inline_comment_prefix: str) -> str:
  """Recursively inline included files and return the inlined text.

  Args:
    path: The path of the file to inline.
    import_root: The root directory to search for included files.
    candidate_includes: Absolute paths to all files which are candidates for
      inclusion.
    include_regex: A regex which matches the programming language's include
      directive.
    inline_comment_prefix: A string prefix used to start a one-line comment in
      the programming language.

  Returns:
    The path with as many included files inlined as possible.
  """
  # TODO(cec): Implement!
  pass
