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
