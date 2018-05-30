"""Unit tests for //deeplearning/clgen/package_util.py."""
import pathlib
import sys

import pytest
from absl import app

from deeplearning.clgen import errors
from deeplearning.clgen import package_util


def test_data_path():
  """Test that data_path() returns a path within //deeplearning/clgen/data."""
  assert pathlib.Path(package_util.data_path('include/opencl-shim.h')).is_file()
  with pytest.raises(errors.File404):
    pathlib.Path(package_util.data_path('not/a/file')).is_file()


def test_pacakge_data():
  """Test that package_data() returns data from //deeplearning/clgen files."""
  assert package_util.package_data('data/include/opencl-shim.h')
  with pytest.raises(errors.File404):
    package_util.package_data("not/a/file")


def test_pacakge_str():
  """Test that package_str() returns str from //deeplearning/clgen files."""
  assert package_util.package_str('data/include/opencl-shim.h')
  with pytest.raises(errors.File404):
    package_util.package_str("not/a/file")


def test_sql_script():
  """package_str() returns script from //deeplearning/clgen/data/sql."""
  assert package_util.sql_script('create-verify-db')
  with pytest.raises(errors.File404):
    package_util.sql_script("not/a/file")


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError('Unrecognized command line flags.')
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
