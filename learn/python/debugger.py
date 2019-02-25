"""Experiments in using ipdb debugger."""
import typing

import ipdb
from absl import app
from absl import flags

FLAGS = flags.FLAGS


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  ipdb.run('x[0] = 3')
  ipdb.set_trace()


if __name__ == '__main__':
  app.run(main)
