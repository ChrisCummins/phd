"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
import typing

from absl import app
from absl import flags

FLAGS = flags.FLAGS


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))


if __name__ == '__main__':
  app.run(main)
