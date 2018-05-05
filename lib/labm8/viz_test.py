"""Unit tests for //lib/labm8:viz."""
import sys

import matplotlib
import pytest
from absl import app

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from lib.labm8 import fs
from lib.labm8 import viz


def _MakeTestPlot():
  t = np.arange(0.0, 2.0, 0.01)
  s = np.sin(2 * np.pi * t)
  plt.plot(t, s)


def test_finalise():
  _MakeTestPlot()
  viz.finalise("/tmp/labm8.png")
  assert fs.exists("/tmp/labm8.png")
  fs.rm("/tmp/labm8.png")


def test_finalise_tight():
  _MakeTestPlot()
  viz.finalise("/tmp/labm8.png", tight=True)
  assert fs.exists("/tmp/labm8.png")
  fs.rm("/tmp/labm8.png")


def test_finalise_figsize():
  _MakeTestPlot()
  viz.finalise("/tmp/labm8.png", figsize=(10, 5))
  assert fs.exists("/tmp/labm8.png")
  fs.rm("/tmp/labm8.png")


def main(argv):  # pylint: disable=missing-docstring
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
