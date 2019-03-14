# Copyright 2017, 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //research/cummins_2017_cgo:generative_model."""
from labm8 import test

from research.cummins_2017_cgo import generative_model


def test_CreateInstanceProtoFromFlags_smoke_test():
  """Test that instance proto can be constructed."""
  assert generative_model.CreateInstanceProtoFromFlags()


def test_CreateInstanceFromFlags_smoke_test():
  """Test that instance can be constructed."""
  assert generative_model.CreateInstanceFromFlags()


if __name__ == '__main__':
  test.Main()
