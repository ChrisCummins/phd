import os
import sys
from io import StringIO

import numpy as np
import pytest
from numpy import testing as nptest
from pathlib import Path
from typing import List

from gpu import cldrive

ENV = cldrive.make_env()


def _platform_name():
  import pyopencl as cl
  _, queue = ENV.ctx_queue()
  device = queue.get_info(cl.command_queue_info.DEVICE)
  platform = device.get_info(cl.device_info.PLATFORM)
  return platform.get_info(cl.platform_info.NAME)


# test decorators
skip_on_pocl = pytest.mark.skipif(
    _platform_name() == "Portable Computing Language",
    reason="not supported on POCL")

needs_cpu = pytest.mark.skipif(
    not cldrive.has_cpu(), reason="no OpenCL CPU device available")

needs_gpu = pytest.mark.skipif(
    not cldrive.has_gpu(), reason="no OpenCL GPU device available")


def data_path(path: str) -> Path:
  return Path(os.path.join(os.path.dirname(__file__), "data", path))


def lol2np(list_of_lists: List[list]) -> np.array:
  return np.array([np.array(x) for x in list_of_lists])


def almost_equal(l1: np.array, l2: np.array) -> None:
  for x, y in zip(l1, l2):
    nptest.assert_almost_equal(lol2np(x), lol2np(y))


class DevNullRedirect(object):
  """
  Context manager to redirect stdout and stderr to devnull.

  Examples
  --------
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
