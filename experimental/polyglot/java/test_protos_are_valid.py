"""Test that //experimental/polyglot/java:protos are valid."""
import pathlib
import sys
import tempfile

import pytest
from absl import app

from datasets.github.scrape_repos.proto import scrape_repos_pb2
from deeplearning.clgen import clgen
from deeplearning.clgen.proto import clgen_pb2
from lib.labm8 import bazelutil
from lib.labm8 import pbutil


def test_clone_list_is_valid():
  """Test that clone_list.pbtxt is valid."""
  pbutil.FromFile(
      bazelutil.DataPath('phd/experimental/polyglot/java/clone_list.pbtxt'),
      scrape_repos_pb2.LanguageCloneList())


def test_config_is_valid():
  """Test that config proto is valid."""
  with tempfile.TemporaryDirectory() as d:
    working_dir = pathlib.Path(d)
    config = pbutil.FromFile(
        bazelutil.DataPath(
            'phd/experimental/polyglot/java/clgen.pbtxt'),
        clgen_pb2.Instance())
    # Change the working directory and corpus path to our bazel run dir.
    config.working_dir = str(working_dir)
    # Make a dummy corpus.
    (working_dir / 'corpus').mkdir()
    (working_dir / 'corpus' / 'foo').touch()
    config.model.corpus.path = str(working_dir / 'corpus')
    clgen.Instance(config)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  app.run(main)
