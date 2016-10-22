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
Text utilities.
"""
from __future__ import division

import labm8 as lab


class Error(Exception):
    """
    Module-level error.
    """
    pass


class TruncateError(Error):
    """
    Thrown in case of truncation error.
    """
    pass


def truncate(string, maxchar):
    """
    Truncate a string to a maximum number of characters.

    If the string is longer than maxchar, then remove excess
    characters and append an ellipses.

    Arguments:

        string (str): String to truncate.
        maxchar (int): Maximum length of string in characters. Must be >= 4.

    Returns:

        str: Of length <= maxchar.

    Raises:

        TruncateError: In case of an error.
    """
    if maxchar < 4:
        raise TruncateError("Maxchar must be > 3")

    if len(string) <= maxchar:
        return string
    else:
        return string[:maxchar - 3] + "..."


def levenshtein(s1, s2):
    """
    Return the Levenshtein distance between two strings.

    Implementation of Levenshtein distance, one of a family of edit
    distance metrics.

    Based on: https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

    Examples:

        >>> text.levensthein("foo", "foo")
        0

        >>> text.levensthein("foo", "fooo")
        1

        >>> text.levensthein("foo", "")
        3

        >>> text.levensthein("1234", "1 34")
        1

    Arguments:

        s1 (str): Argument A.
        s2 (str): Argument B.

    Returns:

        int: Levenshtein distance between the two strings.
    """
    # Left string must be >= right string.
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # Distance is length of s1 if s2 is empty.
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def diff(s1, s2):
    """
    Return a normalised Levenshtein distance between two strings.

    Distance is normalised by dividing the Levenshtein distance of the
    two strings by the max(len(s1), len(s2)).

    Examples:

        >>> text.diff("foo", "foo")
        0

        >>> text.diff("foo", "fooo")
        1

        >>> text.diff("foo", "")
        1

        >>> text.diff("1234", "1 34")
        1

    Arguments:

        s1 (str): Argument A.
        s2 (str): Argument B.

    Returns:

        float: Normalised distance between the two strings.
    """
    return levenshtein(s1, s2) / max(len(s1), len(s2))
