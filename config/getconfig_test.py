"""Unit tests for //config/config.py."""
import os
import pathlib
import sys

import pytest
from absl import app
from absl import flags

from config import getconfig


FLAGS = flags.FLAGS


def test_GetGlobalConfig_system_values():
  """Check that the repo config has all the expected values."""
  config = getconfig.GetGlobalConfig()
  assert config.paths.HasField('repo_root')
  assert config.paths.HasField('cc')
  assert config.paths.HasField('cxx')
  assert config.paths.HasField('python')
  assert config.HasField('with_cuda')


def test_GlobalConfigPaths_repo_root():
  """Test that repo_root is a directory."""
  config = getconfig.GetGlobalConfig()
  assert pathlib.Path(config.paths.repo_root).is_dir()


def test_GlobalConfigPaths_cc():
  """Test that cc is an executable."""
  config = getconfig.GetGlobalConfig()
  assert pathlib.Path(config.paths.cc).is_file()
  assert os.access(config.paths.cc, os.X_OK)


def test_GlobalConfigPaths_cxx():
  """Test that cxx is an executable."""
  config = getconfig.GetGlobalConfig()
  assert pathlib.Path(config.paths.cxx).is_file()
  assert os.access(config.paths.cxx, os.X_OK)


def test_GlobalConfigPaths_python():
  """Test that python is an executable."""
  config = getconfig.GetGlobalConfig()
  assert pathlib.Path(config.paths.python).is_file()
  assert os.access(config.paths.python, os.X_OK)


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
