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
from unittest import main
from tests import TestCase

import math
import labm8 as lab
import labm8.math

class TestMath(TestCase):

    # sqrt() tests
    def test_sqrt_4(self):
        self._test(2, lab.math.sqrt(4))

    def test_sqrt(self):
        self._test(math.sqrt(1024), lab.math.sqrt(1024))

    # mean() tests
    def test_mean_empty_array(self):
        self._test(0, lab.math.mean([]))

    def test_mean_single_item_array(self):
        self._test(1, lab.math.mean([1]))

    def test_mean(self):
        self._test(2, lab.math.mean([1,2,3]))
        self._test((1/3.), lab.math.mean([1,1.5,-1.5]))
        self._test(2, lab.math.mean([2,2,2,2,2]))
        self._test(2.5, lab.math.mean([1,2,3,4]))


    # median() tests
    def test_median_empty_array(self):
        self._test(0, lab.math.median([]))

    def test_median_single_item_array(self):
        self._test(1, lab.math.median([1]))

    def test_median(self):
        self._test(2, lab.math.median([1,2,3]))
        self._test(1, lab.math.median([1,1.5,-1.5]))
        self._test(2.5, lab.math.median([1, 2, 3, 4]))


    # range() tests
    def test_range_empty_array(self):
        self._test(0, lab.math.range([]))

    def test_range_single_item_array(self):
        self._test(0, lab.math.range([1]))

    def test_range_123_array(self):
        self._test(2, lab.math.range([1,2,3]))


    # variance() tests
    def test_variance_empty_array(self):
        self._test(0, lab.math.variance([]))

    def test_variance_single_item_array(self):
        self._test(0, lab.math.variance([1]))

    def test_variance_123_array(self):
        self._test(1, lab.math.variance([1,2,3]))


    # stdev() tests
    def test_stdev_empty_array(self):
        self._test(0, lab.math.stdev([]))

    def test_stdev_single_item_array(self):
        self._test(0, lab.math.stdev([1]))

    def test_stdev_123_array(self):
        self._test(1, lab.math.stdev([1,2,3]))


    # confinterval() tests
    def test_confinterval_empty_array(self):
        self._test((0, 0), lab.math.confinterval([]))

    def test_confinterval_single_item_array(self):
        self._test((1, 1), lab.math.confinterval([1]))

    def test_confinterval_123_array(self):
        self._test((-0.48413771184375287, 4.4841377118437524),
                   lab.math.confinterval([1,2,3]))

    def test_confinterval_all_same(self):
        self._test((1, 1),
                   lab.math.confinterval([1,1,1,1,1]))

    def test_confinterval_c50(self):
        self._test((1.528595479208968, 2.4714045207910322),
                   lab.math.confinterval([1,2,3], conf=0.5))

    def test_confinterval_t_dist(self):
        self._test((0.86841426592382809, 3.1315857340761717),
                   lab.math.confinterval([1,2,3], normal_threshold=1))

if __name__ == '__main__':
    main()
