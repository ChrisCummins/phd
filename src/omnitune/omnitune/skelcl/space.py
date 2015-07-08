from itertools import product

import numpy as np

from . import unhash_params

import labm8 as lab
from labm8 import viz


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
        self.c = wg_c
        self.r = wg_r
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

    def heatmap(self, output=None, title=None, figsize=(5,4), **kwargs):
        import matplotlib.pyplot as plt
        import seaborn as sns

        new_order = list(reversed(range(self.matrix.shape[0])))
        data = self.matrix[:][new_order]

        if "square" not in kwargs:
            kwargs["square"] = True

        _, ax = plt.subplots(figsize=figsize)
        sns.heatmap(data,
                    xticklabels=self.c,
                    yticklabels=list(reversed(self.r)),
                    **kwargs)

        # Set labels.
        ax.set_ylabel("Rows")
        ax.set_xlabel("Columns")
        if title:
            plt.title(title)

        plt.tight_layout()
        plt.gcf().set_size_inches(*figsize, dpi=300)

        viz.finalise(output)

    def trisurf(self, output=None, title=None, figsize=(5,4),
                zlabel=None, zticklabels=None, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        from mpl_toolkits.mplot3d import Axes3D

        num_vals = self.matrix.shape[0] * self.matrix.shape[1]
        X = [0] * num_vals
        Y = [0] * num_vals
        Z = [0] * num_vals

        # Iterate over every point in space.
        for j,i in product(range(self.matrix.shape[0]),
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
        ax.set_xticklabels(self.c)
        ax.set_xlabel("Columns")

        # Set Y axis labels
        ax.set_yticklabels(self.r)
        ax.set_ylabel("Rows")

        # Set Z axis labels
        if zlabel is not None:
            ax.set_zlabel(zlabel)
        if zticklabels is not None:
            ax.set_zticklabels(zlabel)

        plt.tight_layout()
        plt.gcf().set_size_inches(*figsize, dpi=300)
        viz.finalise(output)

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
                c,r = unhash_params(params)
                uniq_c.add(c)
                uniq_r.add(r)
            if wg_c is None:
                wg_c = sorted(list(uniq_c))
            if wg_r is None:
                wg_r = sorted(list(uniq_r))

        space = ParamSpace(wg_c, wg_r)

        for params,value in data.iteritems():
            space[params] = value

        return space
