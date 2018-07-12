"""Unit tests for //deeplearning/deepsmith/difftests/difftests.py."""
import pytest
import sys
from absl import app
from absl import flags

from deeplearning.deepsmith import filters
from deeplearning.deepsmith.proto import deepsmith_pb2


FLAGS = flags.FLAGS

DiffTest = deepsmith_pb2.DifferentialTest
Result = deepsmith_pb2.Result


def test_TODO():
  """Test difftest outcomes when gold standard outcome is unknown."""
  _ = filters


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
