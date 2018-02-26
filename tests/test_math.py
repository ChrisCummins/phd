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
from unittest import main
from tests import TestCase

import math
from labm8 import math as labmath

class TestMath(TestCase):

    # ceil()
    def test_ceil(self):
        self._test(int, type(labmath.ceil(1)))
        self._test(1, labmath.ceil(1))
        self._test(2, labmath.ceil(1.1))
        self._test(3, labmath.ceil(2.5))
        self._test(4, labmath.ceil(3.9))

    def test_ceil_bad_params(self):
        with self.assertRaises(TypeError):
            self._test(None, labmath.ceil(None))
        with self.assertRaises(TypeError):
            self._test(None, labmath.ceil("abc"))

    # floor()
    def test_floor(self):
        self._test(int, type(labmath.floor(1)))
        self._test(1, labmath.floor(1))
        self._test(1, labmath.floor(1.1))
        self._test(2, labmath.floor(2.5))
        self._test(3, labmath.floor(3.9))

    def test_floor_bad_params(self):
        with self.assertRaises(TypeError):
            self._test(None, labmath.floor(None))
        with self.assertRaises(TypeError):
            self._test(None, labmath.floor("abc"))

    # sqrt() tests
    def test_sqrt_4(self):
        self._test(2, labmath.sqrt(4))

    def test_sqrt(self):
        self._test(math.sqrt(1024), labmath.sqrt(1024))

    # mean() tests
    def test_mean_empty_array(self):
        self._test(0, labmath.mean([]))

    def test_mean_single_item_array(self):
        self._test(1, labmath.mean([1]))

    def test_mean(self):
        self._test(2, labmath.mean([1,2,3]))
        self._test((1/3.), labmath.mean([1,1.5,-1.5]))
        self._test(2, labmath.mean([2,2,2,2,2]))
        self._test(2.5, labmath.mean([1,2,3,4]))

    # mean() tests
    def test_geomean_empty_array(self):
        self._test(0, labmath.geomean([]))

    def test_geomean_single_item_array(self):
        self._test(1, labmath.geomean([1]))

    def test_geomean(self):
        self._test(1.81712059283, labmath.geomean([1,2,3]), approximate=True)
        self._test(1.44224957031, labmath.geomean([1,1.5,2]), approximate=True)
        self._test(2, labmath.geomean([2,2,2,2,2]))
        self._test(2.2133638394, labmath.geomean([1,2,3,4]), approximate=True)
        self._test(0, labmath.geomean([0,1,2,3,4]))

    # median() tests
    def test_median_empty_array(self):
        self._test(0, labmath.median([]))

    def test_median_single_item_array(self):
        self._test(1, labmath.median([1]))

    def test_median(self):
        self._test(2, labmath.median([1,2,3]))
        self._test(1, labmath.median([1,1.5,-1.5]))
        self._test(2.5, labmath.median([1, 2, 3, 4]))


    # range() tests
    def test_range_empty_array(self):
        self._test(0, labmath.range([]))

    def test_range_single_item_array(self):
        self._test(0, labmath.range([1]))

    def test_range_123_array(self):
        self._test(2, labmath.range([1,2,3]))


    # variance() tests
    def test_variance_empty_array(self):
        self._test(0, labmath.variance([]))

    def test_variance_single_item_array(self):
        self._test(0, labmath.variance([1]))

    def test_variance_123_array(self):
        self._test(1, labmath.variance([1,2,3]))


    # stdev() tests
    def test_stdev_empty_array(self):
        self._test(0, labmath.stdev([]))

    def test_stdev_single_item_array(self):
        self._test(0, labmath.stdev([1]))

    def test_stdev_123_array(self):
        self._test(1, labmath.stdev([1,2,3]))


    # iqr() tests
    def test_filter_iqr(self):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self._test(0,
                   labmath.iqr(a, 0, 1))
        self._test([4, 5, 6, 7],
                   labmath.iqr(a, 0.25, 0.75))
        self._test([2, 3, 4, 5, 6, 7],
                   labmath.iqr(a, 0.1, 0.75))


    # filter_iqr() tests
    def test_filter_iqr(self):
        a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self._test(a,
                   labmath.filter_iqr(a, 0, 1))
        self._test([4, 5, 6, 7],
                   labmath.filter_iqr(a, 0.25, 0.75))
        self._test([2, 3, 4, 5, 6, 7],
                   labmath.filter_iqr(a, 0.1, 0.75))


    # confinterval() tests
    def test_confinterval_empty_array(self):
        self._test((0, 0), labmath.confinterval([]))

    def test_confinterval_single_item_array(self):
        self._test((1, 1), labmath.confinterval([1]))

    def test_confinterval_123_array(self):
        self._test((-0.48413771184375287, 4.4841377118437524),
                   labmath.confinterval([1,2,3]))

    def test_confinterval_all_same(self):
        self._test((1, 1),
                   labmath.confinterval([1,1,1,1,1]))

    def test_confinterval_c50(self):
        self._test((1.528595479208968, 2.4714045207910322),
                   labmath.confinterval([1,2,3], conf=0.5))

    def test_confinterval_normal_dist(self):
        self._test((0.86841426592382809, 3.1315857340761717),
                   labmath.confinterval([1,2,3], normal_threshold=1))

    def test_confinterval_array_mean(self):
        self._test((1.528595479208968, 2.4714045207910322),
                   labmath.confinterval([1,2,3], conf=0.5, array_mean=2))
        expected_ci = (0.528595479209, 1.47140452079)
        actual_ci = labmath.confinterval([1,2,3], conf=0.5, array_mean=1)
        self._test(expected_ci[0], actual_ci[0], approximate=True)
        self._test(expected_ci[0], actual_ci[0], approximate=True)

    def test_confinterval_error_only(self):
        self._test(0.4714045207910322, labmath.confinterval([1,2,3], conf=.5,
                                                            error_only=True),
                   approximate=True)
