# Copyright (C) 2015-2017 Chris Cummins.
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
from __future__ import division

from unittest import main
from tests import TestCase

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

from labm8 import fs
from labm8 import viz

class TestViz(TestCase):
    def _mkplot(self):
        t = np.arange(0.0, 2.0, 0.01)
        s = np.sin(2*np.pi*t)
        plt.plot(t, s)

    def test_finalise(self):
        self._mkplot()
        viz.finalise("/tmp/labm8.png")
        self.assertTrue(fs.exists("/tmp/labm8.png"))
        fs.rm("/tmp/labm8.png")

    def test_finalise_tight(self):
        self._mkplot()
        viz.finalise("/tmp/labm8.png", tight=True)
        self.assertTrue(fs.exists("/tmp/labm8.png"))
        fs.rm("/tmp/labm8.png")

    def test_finalise_figsize(self):
        self._mkplot()
        viz.finalise("/tmp/labm8.png", figsize=(10, 5))
        self.assertTrue(fs.exists("/tmp/labm8.png"))
        fs.rm("/tmp/labm8.png")
