"""Unit tests for //labm8:viz."""

import pathlib

import matplotlib


matplotlib.use('Agg')

import numpy as np
import pytest
from absl import flags
from matplotlib import pyplot as plt

from labm8 import test
from labm8 import viz


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def test_plot():
  """Test fixture that makes a plot."""
  t = np.arange(0.0, 2.0, 0.01)
  s = np.sin(2 * np.pi * t)
  plt.plot(t, s)


@pytest.mark.parametrize('extension', ('.png', '.jpg', '.pdf'))
def test_Finalize_produces_a_file(test_plot, tempdir: pathlib.Path,
                                  extension: str):
  """"""
  del test_plot
  viz.Finalize(tempdir / f'plot{extension}')
  assert (tempdir / f'plot{extension}').is_file()


@pytest.mark.parametrize('extension', ('.png', '.jpg', '.pdf'))
def test_Finalize_tight(test_plot, tempdir: pathlib.Path, extension: str):
  del test_plot
  viz.Finalize(tempdir / f'plot{extension}', tight=True)
  assert (tempdir / f'plot{extension}').is_file()


@pytest.mark.parametrize('extension', ('.png', '.jpg', '.pdf'))
def test_Finalize_figsize(test_plot, tempdir: pathlib.Path, extension: str):
  del test_plot
  viz.Finalize(tempdir / f'plot{extension}', figsize=(10, 5))
  assert (tempdir / f'plot{extension}').is_file()


if __name__ == '__main__':
  test.Main()
