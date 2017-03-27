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


def _checksum(hash_fn, data):
    return hash_fn(data).hexdigest()


def _checksum_str(hash_fn, string, encoding='utf-8'):
    return _checksum(hash_fn, string.encode(encoding))


def _checksum_list(hash_fn, *elems):
    string = "".join(sorted(str(x) for x in elems))
    return _checksum_str(hash_fn, string)


def _checksum_file(hash_fn, path):
    with open(path, 'rb') as infile:
        ret = _checksum(hash_fn, infile.read())
    return ret


def sha1(data):
    """
    Return the sha1 of "data".

    Arguments:
        data (bytes): Data.

    Returns:
        str: Hex encoded.
    """
    return _checksum(hashlib.sha1, data)


def sha1_str(string, encoding='utf-8'):
    """
    Return the sha1 of string "data".

    Arguments:
        string: String.

    Returns:
        str: Hex encoded.
    """
    return _checksum_str(hashlib.sha1, string, encoding=encoding)


def sha1_list(*elems):
    """
    Return the sha1 of all elements of a list.

    Arguments:
        *elems: List of stringifiable data.

    Returns:
        str: Hex encoded.
    """
    return _checksum_list(hashlib.sha1, *elems)


def sha1_file(path):
    """
    Return the sha1 of file at "path".

    Arguments:
        path (str): Path to file

    Returns:
        str: Hex encoded.
    """
    return _checksum_file(hashlib.sha1, path)


def md5(data):
    """
    Return the md5 of "data".

    Arguments:
        data (bytes): Data.

    Returns:
        str: Hex encoded.
    """
    return _checksum(hashlib.md5, data)


def md5_str(string, encoding='utf-8'):
    """
    Return the md5 of string "data".

    Arguments:
        string: String.

    Returns:
        str: Hex encoded.
    """
    return _checksum_str(hashlib.md5, string, encoding=encoding)


def md5_list(*elems):
    """
    Return the md5 of all elements of a list.

    Arguments:
        *elems: List of stringifiable data.

    Returns:
        str: Hex encoded.
    """
    return _checksum_list(hashlib.md5, *elems)


def md5_file(path):
    """
    Return the md5 of file at "path".

    Arguments:
        path (str): Path to file

    Returns:
        str: Hex encoded.
    """
    return _checksum_file(hashlib.md5, path)


def sha256(data):
    """
    Return the sha256 of "data".

    Arguments:
        data (bytes): Data.

    Returns:
        str: Hex encoded.
    """
    return _checksum(hashlib.sha256, data)


def sha256_str(string, encoding='utf-8'):
    """
    Return the sha256 of string "data".

    Arguments:
        string: String.

    Returns:
        str: Hex encoded.
    """
    return _checksum_str(hashlib.sha256, string, encoding=encoding)


def sha256_list(*elems):
    """
    Return the sha256 of all elements of a list.

    Arguments:
        *elems: List of stringifiable data.

    Returns:
        str: Hex encoded.
    """
    return _checksum_list(hashlib.sha256, *elems)


def sha256_file(path):
    """
    Return the sha256 of file at "path".

    Arguments:
        path (str): Path to file

    Returns:
        str: Hex encoded.
    """
    return _checksum_file(hashlib.sha256, path)
