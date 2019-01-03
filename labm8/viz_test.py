"""Unit tests for //labm8:viz."""

import matplotlib
from absl import flags

from labm8 import test


FLAGS = flags.FLAGS

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from labm8 import fs
from labm8 import viz


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


if __name__ == '__main__':
  test.Main()
