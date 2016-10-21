#
# Copyright 2016 Chris Cummins <chrisc.101@gmail.com>.
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
Deep learning program generator
"""
from __future__ import absolute_import, print_function, with_statement

import json
import os
import re

from hashlib import sha1
from pkg_resources import resource_filename, resource_string, require


class CLgenError(Exception):
    """
    Module error.
    """
    pass


class InternalError(CLgenError):
    """
    An internal module error. This class of errors should not leak outside
    of the module into user code.
    """
    pass


class NotImplementedError(InternalError):
    """
    Code not found.
    """
    pass


class File404(InternalError):
    """
    Data not found.
    """
    pass


def version():
    """
    Get the package version.

    DO NOT try and parse this or derive any special major/minor version
    information from it, treat it as an opaque char array. The only valid
    operators for comparing versions are == and !=.

    Returns:
        str: Version string.
    """
    return require("clgen")[0].version


def must_exist(*path_components, **kwargs):
    """
    Require that a file exists.

    Arguments:
        *path_components (str): Components of the path.
        **kwargs (optional): Key "Error" specifies the exception type to throw.

    Returns:
        str: Path.
    """
    assert(len(path_components))

    path = os.path.expanduser(os.path.join(*path_components))
    if not os.path.exists(path):
        Error = kwargs.get("Error", File404)
        e = Error("path '{}' does not exist".format(path))
        e.path = path
        raise e
    return path


def checksum(data):
    """
    Checksum a byte stream.

    Arguments:
        data (bytes): Data.

    Returns:
        str: Checksum.
    """
    try:
        return sha1(data).hexdigest()
    except Exception:
        raise InternalError("failed to checksum '{}'".format(data[:100]))


def checksum_str(string):
    """
    Checksum a string.

    Arguments:
        string (str): String.

    Returns:
        str: Checksum.
    """
    try:
        return checksum(str(string).encode('utf-8'))
    except UnicodeEncodeError:
        raise InternalError("failed to encode '{}'".format(string[:100]))


def checksum_file(*path_components):
    """
    Checksum a file.

    Arguments:
        path_components (str): Path.

    Returns:
        str: Checksum.
    """
    path = must_exist(*path_components)

    try:
        with open(path) as infile:
            return checksum(infile.read())
    except Exception:
        raise CLgenError("failed to read '{}'".format(path))


def get_substring_idxs(substr, s):
    """
    Return a list of indexes of substr. If substr not found, list is
    empty.

    Arguments:
        substr (str): Substring to match.
        s (str): String to match in.

    Returns:
        list of int: Start indices of substr.
    """
    return [m.start() for m in re.finditer(substr, s)]


def package_path(*path):
    """
    Path to package file.
    """
    path = os.path.expanduser(os.path.join(*path))
    abspath = resource_filename(__name__, path)
    return must_exist(abspath)


def data_path(*path):
    """
    Path to package file.
    """
    return package_path("data", *path)


def package_data(*path):
    """
    Read package data file.

    Arguments:
        path (str): The relative path to the data file, e.g. 'share/foo.txt'.

    Returns:
        bytes: File contents.

    Raises:
        InternalError: In case of IO error.
    """
    path = package_path(*path)

    try:
        return resource_string(__name__, path)
    except Error:
        raise InternalError("failed to read package data '{}'".format(path))


def package_str(*path):
    """
    Read package data file as a string.

    Arguments:
        path (str): The relative path to the text file, e.g. 'share/foo.txt'.

    Returns:
        str: File contents.

    Raises:
        InternalError: In case of IO error.
    """
    try:
        return package_data(*path).decode('utf-8')
    except UnicodeDecodeError:
        raise InternalError("failed to decode package data '{}'".format(path))


def sql_script(name):
    """
    Read SQL script to string.

    Arguments:
        name (str): The name of the SQL script (without file extension).

    Returns:
        str: SQL script.
    """
    path = must_exist('data', 'sql', str(name) + ".sql")
    return package_str(path)


def format_json(data):
    """
    Pretty print JSON.

    Arguments:
        data (dict): JSON blob.
    """
    return json.dumps(data, sort_keys=True, indent=2, separators=(',', ': '))


def main(model_json, arguments_json, sample_json):
    """
    Main entry point for clgen.

    Arguments:
        model_json (dict): Model specification.
        arguments_json (dict): Arguments specification.
        sample_json (dict): Sample specification.
    """
    raise NotImplementedError("coming soon!")
