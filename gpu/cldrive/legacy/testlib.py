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
