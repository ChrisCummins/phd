"""Graphing helper.
"""
from phd.lib.labm8 import io


class Error(Exception):
  """
  Visualisation module base error class.
  """
  pass


def finalise(output=None, figsize=None, tight=True, **kwargs):
  """
  Finalise a plot.

  Display or show the plot, then close it.

  Arguments:

      output (str, optional): Path to save figure to. If not given,
        show plot.
      figsize ((float, float), optional): Figure size in inches.
      **kwargs: Any additional arguments to pass to
        plt.savefig(). Only required if output is not None.
  """
  import matplotlib.pyplot as plt

  # Set figure size.
  if figsize is not None:
    plt.gcf().set_size_inches(*figsize)

  # Set plot layout.
  if tight:
    plt.tight_layout()

  if output is None:
    plt.show()
  else:
    plt.savefig(output, **kwargs)
    io.info("Wrote", output)
  plt.close()


def ShowErrorBarCaps(ax):
  """Show error bar caps.
  
  Seaborn paper style hides error bar caps. Call this function on an axes
  object to make them visible again.
  """
  for ch in ax.get_children():
    if str(ch).startswith('Line2D'):
      ch.set_markeredgewidth(1)
      ch.set_markersize(8)
