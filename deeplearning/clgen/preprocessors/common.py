"""Common preprocessor passes."""
from absl import flags

from deeplearning.clgen import errors


FLAGS = flags.FLAGS


def MinimumLineCount(src: str, min_line_count: int = 3) -> str:
  """Check that file contains a minimum number of lines.

  Args:
    src: The source to verify the line count of.

  Returns:
    src: The unmodified input src.

  Raises:
    NoCodeException: If src is less than min_line_count long.
  """
  if len(src.strip().split('\n')) < min_line_count:
    raise errors.NoCodeException
  return src


def RemoveDuplicateEmptyLines(text: str):
  """Truncate blank lines."""
  last_line = None
  lines = []
  for line in text.split("\n"):
    line = line.rstrip()
    if line or last_line:
      lines.append(line)
    last_line = line
  return "\n".join(lines)
