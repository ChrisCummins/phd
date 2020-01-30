# Copyright (c) 2016-2020 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""Test that //deeplearning/clgen/tests/data/tiny/config.pbtxt is valid."""
import tempfile

from deeplearning.clgen import clgen
from deeplearning.clgen.proto import clgen_pb2
from labm8.py import bazelutil
from labm8.py import pbutil
from labm8.py import test

MODULE_UNDER_TEST = 'deeplearning.clgen'


def test_config_is_valid():
  """Test that config proto is valid."""
  with tempfile.TemporaryDirectory() as d:
    config = pbutil.FromFile(
        bazelutil.DataPath(
            'phd/deeplearning/clgen/tests/data/tiny/config.pbtxt'),
        clgen_pb2.Instance())
    # Change the working directory and corpus path to our bazel run dir.
    config.working_dir = d
    config.model.corpus.local_directory = str(
        bazelutil.DataPath(
            'phd/deeplearning/clgen/tests/data/tiny/corpus.tar.bz2'))
    clgen.Instance(config)


if __name__ == '__main__':
  test.Main()
