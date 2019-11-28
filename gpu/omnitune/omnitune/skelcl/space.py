from itertools import product
from math import log

import numpy as np

from . import hash_params
from . import unhash_params
from labm8.py import io
from labm8.py import viz


class ParamSpace(object):
  """
  Represents the parameter space of workgroup sizes.
  """

  def __init__(self, wg_c, wg_r):
    """
    Create a new parameter space.

    Arguments:

        wg_c (list of int): List of workgroup column values.
        wg_r (list of int): List of workgroup row values.
    """
    self.r = wg_r
    self.c = wg_c
    self.matrix = np.zeros(shape=(len(wg_r), len(wg_c)))

  def wgsize2indexes(self, wgsize):
    wg_c, wg_r = unhash_params(wgsize)
    i = self.c.index(wg_c)
    j = self.r.index(wg_r)
    return j, i

  def __getitem__(self, key):
    return self.get(*self.wgsize2indexes(key))

  def __setitem__(self, key, value):
    j, i = self.wgsize2indexes(key)
    return self.set(j, i, value)

  def __iter__(self):
    return np.nditer(self.matrix)

  def __repr__(self):
    return self.matrix.__repr__()

  def get(self, j, i):
    return self.matrix[j][i]

  def set(self, j, i, value):
    self.matrix[j][i] = value

  def inspace(self, param):
    c, r = unhash_params(param)
    return (c >= min(self.c) and c <= max(self.c) and r >= min(self.r) and
            r <= max(self.r))

  def heatmap(self,
              output=None,
              title=None,
              figsize=(5, 4),
              xlabels=True,
              ylabels=True,
              cbar=True,
              **kwargs):
    import matplotlib.pyplot as plt
    import seaborn as sns

    new_order = list(reversed(range(self.matrix.shape[0])))
    data = self.matrix[:][new_order]

    if "square" not in kwargs:
      kwargs["square"] = True

    if xlabels == True:
      xticklabels = ["" if x % 20 else str(x) for x in self.c]
    else:
      xticklabels = xlabels
    if ylabels == True:
      yticklabels = ["" if x % 20 else str(x) for x in list(reversed(self.r))]
    else:
      yticklabels = ylabels

    _, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data,
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                cbar=cbar,
                **kwargs)

    # Set labels.
    ax.set_ylabel("Rows")
    ax.set_xlabel("Columns")
    if title:
      plt.title(title)

    plt.tight_layout()
    plt.gcf().set_size_inches(*figsize, dpi=300)

    viz.finalise(output)

  def trisurf(self,
              output=None,
              title=None,
              figsize=(5, 4),
              zlabel=None,
              zticklabels=None,
              rotation=None,
              **kwargs):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    num_vals = self.matrix.shape[0] * self.matrix.shape[1]
    if num_vals < 3:
      io.error("Cannot create trisurf of", num_vals, "values")
      return

    X = np.zeros((num_vals,))
    Y = np.zeros((num_vals,))
    Z = np.zeros((num_vals,))

    # Iterate over every point in space.
    for j, i in product(range(self.matrix.shape[0]),
                        range(self.matrix.shape[1])):
      # Convert point to list index.
      index = j * self.matrix.shape[1] + i
      X[index] = i
      Y[index] = j
      Z[index] = self.matrix[j][i]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap=cm.jet, **kwargs)

    # Set X axis labels
    xticks = []
    xticklabels = []
    for i, c in enumerate(self.c):
      if not len(xticks) or c % 20 == 0:
        xticks.append(i)
        xticklabels.append(c)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("$w_c$")

    # Set Y axis labels
    yticks = []
    yticklabels = []
    for i, c in enumerate(self.c):
      if not len(yticks) or c % 20 == 0:
        yticks.append(i)
        yticklabels.append(c)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_ylabel("$w_r$")

    # Set Z axis labels
    if zlabel is not None:
      ax.set_zlabel(zlabel)
    if zticklabels is not None:
      ax.set_zticks(np.arange(len(zticklabels)))
      ax.set_zticklabels(zticklabels)

    # Set plot rotation.
    if rotation is not None: ax.view_init(azim=rotation)
    # Set plot title.
    if title: plt.title(title)
    plt.tight_layout()
    plt.gcf().set_size_inches(*figsize, dpi=300)
    viz.finalise(output)

  def bar3d(self,
            output=None,
            title=None,
            figsize=(5, 4),
            zlabel=None,
            zticklabels=None,
            rotation=None,
            **kwargs):
    import matplotlib.pyplot as plt

    X, Y, dZ = [], [], []

    # Iterate over every point in space.
    for j, i in product(range(self.matrix.shape[0]),
                        range(self.matrix.shape[1])):
      if self.matrix[j][i] > 0:
        X.append(i)
        Y.append(j)
        dZ.append(self.matrix[j][i])

    num_vals = len(X)
    Z = np.zeros((num_vals,))
    dX = np.ones((num_vals,))
    dY = np.ones((num_vals,))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.bar3d(X, Y, Z, dX, dY, dZ, **kwargs)

    # Set X axis labels
    ax.set_xticks(np.arange(len(self.c)))
    ax.set_xticklabels(self.c)
    ax.set_xlabel("Columns")

    # Set Y axis labels
    ax.set_yticks(np.arange(len(self.r)))
    ax.set_yticklabels(self.r)
    ax.set_ylabel("Rows")

    # Set Z axis labels
    if zlabel is not None:
      ax.set_zlabel(zlabel)
    if zticklabels is not None:
      ax.set_zticks(np.arange(len(zticklabels)))
      ax.set_zticklabels(zticklabels)

    # Set plot rotation.
    if rotation is not None: ax.view_init(azim=rotation)
    # Set plot title.
    if title: plt.title(title)
    plt.tight_layout()
    plt.gcf().set_size_inches(*figsize, dpi=300)
    viz.finalise(output)

  def reshape(self, max_c=0, max_r=0, min_c=0, min_r=0):
    if max_c < 1:
      max_c = len(self.c) + max_c
    if max_r < 1:
      max_r = len(self.r) + max_r

    new_r = self.r[min_r:max_r]
    new_c = self.c[min_c:max_c]
    new_matrix = np.zeros(shape=(len(new_r), len(new_c)))

    for j in range(min_r, max_r):
      for i in range(min_c, max_c):
        new_matrix[j][i] = self.matrix[j - min_r][i - min_c]

    self.c = new_c
    self.r = new_r
    self.matrix = new_matrix

  def log(self):
    for j in range(len(self.r)):
      for i in range(len(self.c)):
        if self.matrix[j][i] > 0:
          self.matrix[j][i] = log(self.matrix[j][i])

  def clip(self, max_c=0, max_r=0):
    """
    Clip a space within a bounded area.
    """
    new_c = min(len(self.c), max_c)
    new_r = min(len(self.r), max_r)
    self.reshape(new_c, new_r)

  @staticmethod
  def from_dict(data, wg_c=None, wg_r=None):
    """
    Construct a parameter space from a dict.

    Determines the bounding space from the dict, then populates
    the space with the values from the dict.

    Arguments:

        data (dict of {str: <type>} pairs): Data to create space from.
        wg_c (list of int, optional): Legal wg_c values. If not
          given, values are deduced automatically from the data.
        wg_r (list of int, optional): Legal wg_r values. If not
          given, values are deduced automatically from the data.

    Returns:

        ParamSpace: Populated param space.
    """
    if wg_c is None or wg_r is None:
      uniq_c, uniq_r = set(), set()
      for params in data.keys():
        c, r = unhash_params(params)
        uniq_c.add(c)
        uniq_r.add(r)
      if wg_c is None:
        wg_c = sorted(list(uniq_c))
      if wg_r is None:
        wg_r = sorted(list(uniq_r))

    space = ParamSpace(wg_c, wg_r)

    for params, value in data.iteritems():
      space[params] = value

    return space


def enumerate_wlegal_params(maxwgsize):
  return [
      hash_params(j, i)
      for j, i in product(range(2, maxwgsize / 2 +
                                1, 2), range(2, maxwgsize / 2 + 1, 2))
      if j * i < maxwgsize
  ]
