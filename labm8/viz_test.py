"""Unit tests for //labm8:viz."""

import pathlib

import matplotlib
import pandas as pd

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


@pytest.mark.parametrize('extension', ('.png', '.pdf'))
def test_Finalize_produces_a_file(test_plot, tempdir: pathlib.Path,
                                  extension: str):
  """That file is produced."""
  del test_plot
  viz.Finalize(tempdir / f'plot{extension}')
  assert (tempdir / f'plot{extension}').is_file()


@pytest.mark.parametrize('extension', ('.png', '.pdf'))
def test_Finalize_tight(test_plot, tempdir: pathlib.Path, extension: str):
  """That tight keyword."""
  del test_plot
  viz.Finalize(tempdir / f'plot{extension}', tight=True)
  assert (tempdir / f'plot{extension}').is_file()


@pytest.mark.parametrize('extension', ('.png', '.pdf'))
def test_Finalize_figsize(test_plot, tempdir: pathlib.Path, extension: str):
  """That figsize keyword."""
  del test_plot
  viz.Finalize(tempdir / f'plot{extension}', figsize=(10, 5))
  assert (tempdir / f'plot{extension}').is_file()


def test_Distplot_dataframe():
  """Test plotting dataframe."""
  df = pd.DataFrame({'x': 1, 'group': 'foo'}, {'x': 2, 'group': 'bar'})
  viz.Distplot(x='x', hue='group', data=df)


def test_Distplot_with_hue_order():
  """Test plotting with hue order."""
  df = pd.DataFrame({'x': 1, 'group': 'foo'}, {'x': 2, 'group': 'bar'})
  viz.Distplot(x='x', hue='group', hue_order=['foo', 'bar'], data=df)


def test_Distplot_with_missing_hue_order_values():
  """Plotting with missing hue order is not an error."""
  df = pd.DataFrame({'x': 1, 'group': 'foo'}, {'x': 2, 'group': 'bar'})
  viz.Distplot(x='x', hue='group', hue_order=['foo', 'bar', 'car'], data=df)


if __name__ == '__main__':
  test.Main()
