"""Unit tests for //learn/daily/d181212_graph_nets_shortest_path:model."""
import sys
import typing

import pytest
from absl import app
from absl import flags
from learn.daily.d181212_graph_nets_shortest_path import model


FLAGS = flags.FLAGS


def test_TODO():
  """Short summary of test."""
  _ = model


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
