"""Simple demo using ascii art package."""
import sys

import ascii_art
import pytest
from absl import app
from absl import flags


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


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
