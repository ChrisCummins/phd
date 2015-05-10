# Copyright (C) 2015 Chris Cummins.
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
from unittest import TestCase, main

import math
import labm8 as lab
import labm8.math

class TestMath(TestCase):

    # sqrt() tests
    def test_sqrt_4(self):
        self.assertTrue(lab.math.sqrt(4) == 2)

    def test_sqrt(self):
        self.assertTrue(lab.math.sqrt(1024) == math.sqrt(1024))

    # mean() tests
    def test_mean_empty_array(self):
        data = []
        self.assertTrue(lab.math.mean(data) == 0)

    def test_mean_single_item_array(self):
        data = [1]
        self.assertTrue(lab.math.mean(data) == 1)

    def test_mean_123_array(self):
        data = [1, 2, 3]
        self.assertTrue(lab.math.mean(data) == 2)


    # range() tests
    def test_range_empty_array(self):
        data = []
        self.assertTrue(lab.math.range(data) == 0)

    def test_range_single_item_array(self):
        data = [1]
        self.assertTrue(lab.math.range(data) == 0)

    def test_range_123_array(self):
        data = [1, 2, 3]
        self.assertTrue(lab.math.range(data) == 2)


    # variance() tests
    def test_variance_empty_array(self):
        data = []
        self.assertTrue(lab.math.variance(data) == 0)

    def test_variance_single_item_array(self):
        data = [1]
        self.assertTrue(lab.math.variance(data) == 0)

    def test_variance_123_array(self):
        data = [1, 2, 3]
        self.assertTrue(lab.math.variance(data) == 1)


    # stdev() tests
    def test_stdev_empty_array(self):
        data = []
        self.assertTrue(lab.math.stdev(data) == 0)

    def test_stdev_single_item_array(self):
        data = [1]
        self.assertTrue(lab.math.stdev(data) == 0)

    def test_stdev_123_array(self):
        data = [1, 2, 3]
        self.assertTrue(lab.math.stdev(data) == 1)


    # confinterval() tests
    def test_confinterval_empty_array(self):
        data = []
        self.assertTrue(lab.math.confinterval(data) == (0, 0))

    def test_confinterval_single_item_array(self):
        data = [1]
        self.assertTrue(lab.math.confinterval(data) == (1, 1))

    def test_confinterval_123_array(self):
        data = [1, 2, 3]
        self.assertTrue(lab.math.confinterval(data) ==
                        (-0.48413771184375287, 4.4841377118437524))

    def test_confinterval_c50(self):
        data = [1, 2, 3]
        self.assertTrue(lab.math.confinterval(data, conf=0.5) ==
                        (1.528595479208968, 2.4714045207910322))

    def test_confinterval_t_dist(self):
        data = [1, 2, 3]
        self.assertTrue(lab.math.confinterval(data, normal_threshold=1) ==
                        (0.86841426592382809, 3.1315857340761717))

if __name__ == '__main__':
    main()
