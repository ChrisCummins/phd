# Copyright (C) 2015, 2016 Chris Cummins.
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
Python type utilities.
"""
from collections import Mapping
from six import string_types

def is_str(s):
    """
    Return whether variable is string type.

    On python 3, unicode encoding is *not* string type. On python 2, it is.

    Arguments:
        s: Value.

    Returns:
        bool: True if is string, else false.
    """
    return isinstance(s, string_types)


def is_dict(obj):
    """
    Check if an object is a dict.
    """
    return isinstance(obj, dict)


def is_seq(obj):
    """
    Check if an object is a sequence.
    """
    return (not is_str(obj) and not is_dict(obj) and
            (hasattr(obj, "__getitem__") or hasattr(obj, "__iter__")))


def flatten(lists):
    """
    Flatten a list of lists.
    """
    return [item for sublist in lists for item in sublist]


def update(dst, src):
    """
    Recursively update values in dst from src.

    Unlike the builtin dict.update() function, this method will decend into
    nested dicts, updating all nested values.

    Arguments:
        dst (dict): Destination dict.
        src (dict): Source dict.

    Returns:
        dict: dst updated with entries from src.
    """
    for k, v in src.items():
        if isinstance(v, Mapping):
            r = update(dst.get(k, {}), v)
            dst[k] = r
        else:
            dst[k] = src[k]
    return dst


def dict_values(src):
    """
    Recursively get values in dict.

    Unlike the builtin dict.values() function, this method will descend into
    nested dicts, returning all nested values.

    Arguments:
        src (dict): Source dict.

    Returns:
        list: List of values.
    """
    for v in src.values():
        if isinstance(v, dict):
            for v in dict_values(v):
                yield v
        else:
            yield v
