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
Time utilities.
"""
import datetime

DEFAULT_DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"


def strfmt(datetime, format=DEFAULT_DATETIME_FORMAT):
    """
    Format date to string.
    """
    return datetime.strftime(format)


def now():
    """
    Get the current datetime.
    """
    return datetime.datetime.now()


def nowstr(format=DEFAULT_DATETIME_FORMAT):
    """
    Convenience wrapper to get the current time as a string.

    Equivalent to invoking strfmt(now()).
    """
    return strfmt(now(), format=format)
