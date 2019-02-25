"""Utility code for working with shells."""

from absl import flags

FLAGS = flags.FLAGS


class ShellEscapeCodes(object):
  """Shell escape codes for pretty-printing."""
  PURPLE = "\033[95m"
  CYAN = "\033[96m"
  DARKCYAN = "\033[36m"
  BLUE = "\033[94m"
  GREEN = "\033[92m"
  YELLOW = "\033[93m"
  RED = "\033[91m"
  BOLD = "\033[1m"
  UNDERLINE = "\033[4m"
  END = "\033[0m"
