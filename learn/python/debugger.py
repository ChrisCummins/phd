"""Experiments in using ipdb debugger."""
import typing

import ipdb

from labm8 import app

FLAGS = app.FLAGS


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  ipdb.run('x[0] = 3')
  ipdb.set_trace()


if __name__ == '__main__':
  app.RunWithArgs(main)
