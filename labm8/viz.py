"""Plotting and visualization utilities."""
import pathlib
import typing

from absl import flags
from absl import logging
from matplotlib import axes
from matplotlib import pyplot as plt


FLAGS = flags.FLAGS


def Finalize(output: typing.Optional[typing.Union[str, pathlib.Path]] = None,
             figsize=None, tight=True, **savefig_opts):
  """Finalise a plot.

  Display or show the plot, then close it.

  Args:
    output: Path to save figure to. If not given, plot is shown.
    figsize: Figure size in inches.
    **savefig_opts: Any additional arguments to pass to
      plt.savefig(). Only required if output is not None.
  """
  # Set figure size.
  if figsize is not None:
    plt.gcf().set_size_inches(*figsize)

  # Set plot layout.
  if tight:
    plt.tight_layout()

  if output is None:
    plt.show()
  else:
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output), **savefig_opts)
    logging.info("Wrote '%s'", output)
  plt.close()


def ShowErrorBarCaps(ax: axes.Axes):
  """Show error bar caps.
  
  Seaborn paper style hides error bar caps. Call this function on an axes
  object to make them visible again.
  """
  for child in ax.get_children():
    if str(child).startswith('Line2D'):
      child.set_markeredgewidth(1)
      child.set_markersize(8)


def RotateXLabels(rotation: int = 90, ax: axes.Axes = None) -> None:
  """Rotate plot X labels anti-clockwise.

  Args:
    rotation: The number of degrees to rotate the labels by.
    ax: The plot axis.
  """
  ax = ax or plt.gca()
  plt.setp(ax.get_xticklabels(), rotation=rotation)


def RotateYLabels(rotation: int = 90, ax: axes.Axes = None):
  """Rotate plot Y labels anti-clockwise.

  Args:
    rotation: The number of degrees to rotate the labels by.
    ax: The plot axis.
  """
  ax = ax or plt.gca()
  plt.setp(ax.get_yticklabels(), rotation=rotation)
