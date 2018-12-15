"""Unit tests for //experimental/compilers/reachability:llvm_util."""
import pathlib
import sys

import pytest
from absl import app
from absl import flags

from experimental.compilers.reachability import llvm_util


FLAGS = flags.FLAGS


def test_ControlFlowGraphFromBytecode_TODO():
  """TODO"""
  # TODO(cec): Create a bytecode file, test control flow graph.
  llvm_util.ControlFlowGraphFromBytecode(pathlib.Path('/dev/null'))


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
