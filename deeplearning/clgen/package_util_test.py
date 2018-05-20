#
# Copyright 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
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
