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
Hashing and cryptography utils.
"""
import hashlib


def _sha1(data):
    return hashlib.sha1(data).hexdigest()


def sha1(data):
    """
    Return the sha1 of string "data".
    """
    return _sha1(data.encode("utf-8"))


def sha1_file(path):
    """
    Return the sha1 of file at "path".
    """
    with open(path, 'rb') as infile:
        sha1 = _sha1(infile.read())
    return sha1
