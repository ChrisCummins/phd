# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
# This file is part of cldrive.
#
# cldrive is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cldrive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //gpu/cldrive/legacy/env.py."""

import pytest
from absl import flags

from gpu.cldrive.legacy import env
from labm8 import test

FLAGS = flags.FLAGS


def test_OclgrindOpenCLEnvironment_Exec_version():
  """Test that OclgrindOpenCLEnvironment.Exec() works as expected."""
  proc = env.OclgrindOpenCLEnvironment().Exec(['--version'])
  # This test will of course fail if the @oclgrind package is updated.
  assert proc.stdout == """
Oclgrind 18.3

Copyright (c) 2013-2018
James Price and Simon McIntosh-Smith, University of Bristol
https://github.com/jrprice/Oclgrind

"""


def test_OclgrindOpenCLEnvironment_name():
  """Test that the OclgrindOpenCLEnvironment has a correct 'name' property."""
  env_ = env.OclgrindOpenCLEnvironment()
  # This test will of course fail if the @oclgrind package is updated.
  assert 'Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2' == env_.name


def test_OcldringOpenCLEnvironment_FromName_found():
  """Test name that can be found."""
  env_ = env.OclgrindOpenCLEnvironment.FromName(
      'Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2')
  assert env_.name == 'Emulator|Oclgrind|Oclgrind_Simulator|Oclgrind_18.3|1.2'


def test_OcldringOpenCLEnvironment_FromName_not_found():
  """Test name that can't be found."""
  with pytest.raises(LookupError):
    env.OclgrindOpenCLEnvironment.FromName('Not a real environment')


if __name__ == "__main__":
  test.Main()
