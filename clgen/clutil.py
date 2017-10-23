#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
"""
OpenCL utilities.
"""
import numpy as np
import re

from labm8 import text
from six import string_types
from typing import List, Tuple


def get_attribute_range(src: str, start_idx: int) -> Tuple[int, int]:
    """
    Get string indices range of attributes.

    Parameters
    ----------
    src : str
        OpenCL kernel source.
    start_idx : int
        Index of attribute opening brace.

    Returns
    -------
    Tuple[int, int]
        Start and end indices of attributes.
    """
    i = src.find('(', start_idx) + 1
    d = 1
    while i < len(src) and d > 0:
        if src[i] == '(':
            d += 1
        elif src[i] == ')':
            d -= 1
        i += 1

    return (start_idx, i)


def strip_attributes(src: str) -> str:
    """
    Remove attributes from OpenCL source.

    Parameters
    ----------
    src : str
        OpenCL source.

    Returns
    -------
    str
        OpenCL source, with ((attributes)) removed.
    """
    # get list of __attribute__ substrings
    idxs = sorted(text.get_substring_idxs('__attribute__', src))

    # get ((attribute)) ranges
    attribute_ranges = [get_attribute_range(src, i) for i in idxs]

    # remove ((atribute)) ranges
    for r in reversed(attribute_ranges):
        src = src[:r[0]] + src[r[1]:]
    return src
