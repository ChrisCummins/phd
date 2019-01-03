"""Simple demo using ascii art package."""

import ascii_art
from absl import flags

from labm8 import test


FLAGS = flags.FLAGS


def MultiLineRightStrip(s):
  """Right strip all lines of a multi-line string."""
  return '\n'.join(x.rstrip() for x in str(s).split('\n'))


def test_Chart():
  """Test that Chart produces text."""
  data = [
    5.2,
    2.1,
    2.3,
    -10,
  ]
  c = ascii_art.Chart(
      data,
      width=30, height=20, padding=2, axis_char=u'|'
  )

  assert MultiLineRightStrip(c.render()) == """
  5.2 |       ░
      | █     ░
      | █     ░
      | █     ░
      | █     ░
      | █     ░
      | █     ░
      | █     ░
      | █     ░
      | █ █ █ ░
      | █ █ █ ░
      | █ █ █ ░
      | █ █ █ ░
      | █ █ █ ░
      | █ █ █ ░
    0 . . . . ░ . . . . .
"""


if __name__ == '__main__':
  test.Main()
