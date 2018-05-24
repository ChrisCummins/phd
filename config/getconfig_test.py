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
  """Check that the repo config has all of the expected fields."""
  config = getconfig.GetGlobalConfig()
  assert config.paths.HasField('repo_root')
  assert config.paths.HasField('cc')
  assert config.paths.HasField('cxx')
  assert config.paths.HasField('opt')
  assert config.paths.HasField('libclang_so')
  assert config.paths.HasField('clang_format')
  assert config.paths.HasField('python')
  assert config.paths.HasField('llvm_prefix')
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


def test_GlobalConfigPaths_opt():
  """Test that opt is an executable."""
  config = getconfig.GetGlobalConfig()
  assert pathlib.Path(config.paths.opt).is_file()
  assert os.access(config.paths.opt, os.X_OK)


def test_GlobalConfigPaths_libclang_so():
  """Test that libclang_so is a file, if set."""
  config = getconfig.GetGlobalConfig()
  if config.paths.libclang_so:
    assert pathlib.Path(config.paths.libclang_so).is_file()


def test_GlobalConfigPaths_clang_format():
  """Test that clang-format is an executable."""
  config = getconfig.GetGlobalConfig()
  assert pathlib.Path(config.paths.clang_format).is_file()
  assert os.access(config.paths.clang_format, os.X_OK)


def test_GlobalConfigPaths_python():
  """Test that python is an executable."""
  config = getconfig.GetGlobalConfig()
  assert pathlib.Path(config.paths.python).is_file()
  assert os.access(config.paths.python, os.X_OK)


def test_GlobalConfigPaths_llvm_prefix():
  """Test that llvm_prefix is a directory."""
  config = getconfig.GetGlobalConfig()
  assert pathlib.Path(config.paths.llvm_prefix).is_dir()
  assert (pathlib.Path(
    config.paths.llvm_prefix) / 'bin' / 'llvm-config').is_file()


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
