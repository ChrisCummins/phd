# Copyright (C) 2015, 2016 Chris Cummins.
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
#
"""
Math utils. Import as "labmath" to prevent conflicts with system math package.
"""
# Use absolute paths for imports so as to prevent a conflict with the
# system "math" module.
from __future__ import absolute_import

# Use floating point division by default.
from __future__ import division

import math
import sys

import numpy as np
import scipy
from scipy import stats

import labm8 as lab
from labm8 import modules

if lab.is_python3():
    from functools import reduce


def ceil(number):
    """
    Return the ceiling of a number as an int.

    This is the smallest integral value >= number.

    Example:

        >>> labmath.ceil(1.5)
        2

    Arguments:

        number (float): A numeric value.

    Returns:

        int: Smallest integer >= number.

    Raises:

        TypeError: If argument is not a numeric value.
    """
    return int(math.ceil(number))


def floor(number):
    """
    Return the floor of a number as an int.

    This is the largest integral value <= number.

    Example:

        >>> labmath.floor(1.5)
        1

    Arguments:

        number (float): A numeric value.

    Returns:

        int: Largest integer <= number.

    Raises:

        TypeError: If argument is not a numeric value.
    """
    return int(math.floor(number))


def sqrt(number):
    """
    Return the square root of a number.
    """
    return math.sqrt(number)


def mean(array):
    """
    Return the mean value of a list of divisible numbers.
    """
    n = len(array)

    if n < 1:
        return 0
    elif n == 1:
        return array[0]
    return sum(array) / n


def geomean(array):
    """
    Return the mean value of a list of divisible numbers.
    """
    n = len(array)

    if n < 1:
        return 0
    elif n == 1:
        return array[0]
    return stats.mstats.gmean(array)


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


def iqr(array, lower, upper):
    """
    Return interquartile range for given array.

    Arguments:

        lower (float): Lower bound for IQR, in range 0 <= lower <= 1.
        upper (float): Upper bound for IQR, in range 0 <= upper <= 1.

    Returns:

        (float, float): Lower and upper IQR values.
    """
    return np.percentile(array, [lower * 100, upper * 100])


def filter_iqr(array, lower, upper):
    """
    Return elements which falls within specified interquartile range.

    Arguments:

        array (list): Sequence of numbers.
        lower (float): Lower bound for IQR, in range 0 <= lower <= 1.
        upper (float): Upper bound for IQR, in range 0 <= upper <= 1.

    Returns:

        list: Copy of original list, with elements outside of IQR
          removed.
    """
    upper, lower = iqr(array, upper, lower)

    new = list(array)
    for x in new[:]:
        if x < lower or x > upper:
            new.remove(x)

    return new


def confinterval(array, conf=0.95, normal_threshold=30, error_only=False,
                 array_mean=None):
    """
    Return the confidence interval of a list for a given confidence.

    Arguments:

        array (list): Sequence of numbers.
        conf (float): Confidence interval, in range 0 <= ci <= 1
        normal_threshold (int): The number of elements in the array is
          < normal_threshold, use a T-distribution. If the number of
          elements in the array is >= normal_threshold, use a normal
          distribution.
        error_only (bool, optional): If true, return only the size of
          symmetrical confidence interval, equal to ci_upper - mean.
        array_mean (float, optional): Optimisation trick for if you
          already know the arithmetic mean of the array to prevent
          this function from re-calculating.

    Returns:

        (float, float): Lower and upper bounds on confidence interval,
          respectively.
    """
    n = len(array)

    if n < 1:
        # We have no data.
        array_mean, c0, c1 = 0, 0, 0
    elif n == 1:
        # We have only a single datapoint, so return that value.
        array_mean, c0, c1 = array[0], array[0], array[0]
    else:
        scale = stdev(array) / sqrt(n)
        # Check if all values are the same.
        values_all_same = all(x == array[0] for x in array[1:])

        if values_all_same:
            # If values are all the same, return that value.
            array_mean, c0, c1 = array[0], array[0], array[0]
        else:
            if array_mean is None: array_mean = mean(array)
            if n < normal_threshold:
                # We have a "small" number of datapoints, so use a
                # t-distribution.
                c0, c1 = scipy.stats.t.interval(conf, n - 1, loc=array_mean,
                                                scale=scale)
            else:
                # We have a "large" number of datapoints, so use a
                # normal distribution.
                c0, c1 = scipy.stats.norm.interval(conf, loc=array_mean,
                                                   scale=scale)

    if error_only:
        return c1 - array_mean
    else:
        return c0, c1
