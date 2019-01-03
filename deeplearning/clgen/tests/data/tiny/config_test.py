"""Test that //deeplearning/clgen/tests/data/tiny/config.pbtxt is valid."""
import tempfile

from deeplearning.clgen import clgen
from deeplearning.clgen.proto import clgen_pb2
from labm8 import bazelutil
from labm8 import pbutil
from labm8 import test


def test_config_is_valid():
  """Test that config proto is valid."""
  with tempfile.TemporaryDirectory() as d:
    config = pbutil.FromFile(
        bazelutil.DataPath(
            'phd/deeplearning/clgen/tests/data/tiny/config.pbtxt'),
        clgen_pb2.Instance())
    # Change the working directory and corpus path to our bazel run dir.
    config.working_dir = d
    config.model.corpus.local_directory = str(bazelutil.DataPath(
        'phd/deeplearning/clgen/tests/data/tiny/corpus.tar.bz2'))
    clgen.Instance(config)


if __name__ == '__main__':
  test.Main()
