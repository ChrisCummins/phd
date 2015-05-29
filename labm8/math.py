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
import sys

import numpy
import scipy
from scipy import stats

import labm8 as lab
from labm8 import modules

# Namespace collision between this "math" and system "math" packages.
if lab.is_python3():
    import math as std_math
else:
    std_math = modules.import_foreign("math", "std_math")

def sqrt(number):
    """
    Return the square root of a number.
    """
    return std_math.sqrt(number)

def mean(array):
    """
    Return the mean value of a list of divisible numbers.
    """
    n = len(array)

    if n < 1:
        return 0
    elif n == 1:
        return array[0]
    return sum([float(x) for x in array]) / float(n)


def median(array):
    """
    Return the median value of a list of numbers.
    """
    n = len(array)

    if n < 1:
        return 0
    elif n == 1:
        return array[0]

    sorted_vals = sorted(array)
    midpoint = int(n / 2)
    if n % 2 == 1:
        return sorted_vals[midpoint]
    else:
        return (sorted_vals[midpoint - 1] + sorted_vals[midpoint]) / 2.0


def range(array):
    """
    Return the range between min and max values.
    """
    if len(array) < 1:
        return 0
    return max(array) - min(array)

def variance(array):
    """
    Return the variance of a list of divisible numbers.
    """
    if len(array) < 2:
        return 0
    u = mean(array)
    return sum([(x - u) ** 2 for x in array]) / (len(array) - 1)

def stdev(array):
    """
    Return the standard deviation of a list of divisible numbers.
    """
    return sqrt(variance(array))

def confinterval(array, conf=0.95, normal_threshold=30):
    """
    Return the confidence interval of a list for a given confidence.
    """
    n = len(array)

    if n < 1:
        # We have no data.
        return (0, 0)
    elif n == 1:
        # We have only a single datapoint, so return that value.
        return (array[0], array[0])

    scale = stdev(array) / sqrt(n)
    # Check if all values are the same.
    values_all_same = all(x == array[0] for x in array[1:])

    if values_all_same:
        # If values are all the same, return that value.
        return (array[0], array[0])
    if n < normal_threshold:
        # We have a "small" number of datapoints, so use a t-distribution.
        return scipy.stats.t.interval(conf, n - 1, loc=mean(array), scale=scale)
    else:
        # We have a "large" number of datapoints, so use a normal distribution.
        return scipy.stats.norm.interval(conf, loc=mean(array), scale=scale)
