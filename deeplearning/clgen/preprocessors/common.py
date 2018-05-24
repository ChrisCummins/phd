"""Common preprocessor passes."""
from absl import flags

from deeplearning.clgen import errors
from deeplearning.clgen.preprocessors import preprocessors


FLAGS = flags.FLAGS


def _MinimumLineCount(text: str, min_line_count: int) -> str:
  """Private implementation of minimum number of lines.

  Args:
    text: The source to verify the line count of.

  Returns:
    src: The unmodified input src.

  Raises:
    NoCodeException: If src is less than min_line_count long.
  """
  if len(text.strip().split('\n')) < min_line_count:
    raise errors.NoCodeException
  return text


@preprocessors.clgen_preprocessor
def MinimumLineCount3(text: str) -> str:
  """Check that file contains a minimum number of lines.

  Args:
    text: The source to verify the line count of.

  Returns:
    src: The unmodified input src.

  Raises:
    NoCodeException: If src is less than min_line_count long.
  """
  return _MinimumLineCount(text, 3)


@preprocessors.clgen_preprocessor
def StripDuplicateEmptyLines(text: str):
  """Truncate blank lines."""
  last_line = None
  lines = []
  for line in text.split("\n"):
    line = line.rstrip()
    if line or last_line:
      lines.append(line)
    last_line = line
  return "\n".join(lines)
