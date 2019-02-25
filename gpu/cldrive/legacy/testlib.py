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
"""Shared testing utilities."""
import typing

import numpy as np
from numpy import testing as nptest


def ListOfListsToNumpy(list_of_lists: typing.List[list]) -> np.array:
  """Convert list of lists to 2D numpy array."""
  return np.array([np.array(x) for x in list_of_lists])


def Assert2DArraysAlmostEqual(l1: np.array, l2: np.array) -> None:
  """Assert that 2D arrays are almost equal."""
  for x, y in zip(l1, l2):
    nptest.assert_almost_equal(ListOfListsToNumpy(x), ListOfListsToNumpy(y))


class DevNullRedirect(object):
  """Context manager to redirect stdout and stderr to devnull.

  Examples:
    >>> with DevNullRedirect(): print("this will not print")
  """

  def __init__(self):
    self.stdout = None
    self.stderr = None

  def __enter__(self):
    self.stdout = sys.stdout
    self.stderr = sys.stderr

    sys.stdout = StringIO()
    sys.stderr = StringIO()

  def __exit__(self, *args):
    sys.stdout = self.stdout
    sys.stderr = self.stderr
