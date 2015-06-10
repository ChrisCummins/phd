import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

class ParamSpace(object):
    def __init__(self, wg_c, wg_r):
        self.c = wg_c
        self.r = wg_r
        self.matrix = np.zeros(shape=(len(wg_c), len(wg_r)))

    def wgsize2indexes(self, wgsize):
        wg_c, wg_r = wgsize.split("x")
        wg_c, wg_r = int(wg_c), int(wg_r)
        i = self.c.index(wg_c)
        j = self.r.index(wg_r)
        return j, i

    def __getitem__(self, key):
        j, i = self.wgsize2indexes(key)
        return self.matrix[j][i]

    def __setitem__(self, key, value):
        j, i = self.wgsize2indexes(key)
        self.matrix[j][i] = value

    def heatmap(self, path=None, figsize=(5,4), **kwargs):
        new_order = list(reversed(range(self.matrix.shape[0])))
        data = self.matrix[:][new_order]

        _, ax = plt.subplots(figsize=figsize)
        sns.heatmap(data,
                    xticklabels=self.c,
                    yticklabels=list(reversed(self.r)),
                    **kwargs)

        # Set labels.
        ax.set_ylabel("Rows")
        ax.set_xlabel("Columns")

        plt.tight_layout()
        plt.gcf().set_size_inches(*figsize, dpi=300)

        if path:
            plt.savefig(path)
        else:
            plt.show()
        plt.close()
