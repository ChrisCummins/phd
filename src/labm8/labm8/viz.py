# Copyright (C) 2015 Chris Cummins.
#
# This file is part of labm8.
#
# Labm8 is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Labm8 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with labm8.  If not, see <http://www.gnu.org/licenses/>.
"""
Visualisations module.
"""
import labm8 as lab
from labm8 import io


def finalise(output=None):
    """
    Finalise a plot.

    Display or show the plot, then close it.

    Arguments:

        output (str, optional): Path to save figure to. If not given,
          show plot.
    """
    import matplotlib.pyplot as plt

    if output is None:
        plt.show()
    else:
        plt.savefig(output)
        io.info("Wrote '{}'".format(output))
    plt.close()
