from __future__ import absolute_import
from __future__ import print_function
from __future__ import with_statement

import csv
import json
import re
import os

from hashlib import sha1
from pkg_resources import resource_filename,resource_string

import labm8
from labm8 import fs
from io import open


class SmithException(Exception): pass
class InternalException(SmithException): pass
class NotImplementedException(InternalException): pass
class Data404Exception(InternalException): pass


def assert_exists(*path_components, **kwargs):
    path = fs.path(*path_components)
    if not os.path.exists(path):
        exception = kwargs.get("exception", SmithException)
        raise exception("path '{}' does not exist".format(path))
    return path


def checksum(data):
    try:
        return sha1(data).hexdigest()
    except Exception:
        raise InternalException("failed to hash '{}'".format(data[:100]))


def checksum_str(string):
    try:
        return checksum(str(string).encode('utf-8'))
    except UnicodeEncodeError:
        raise InternalException("failed to encode '{}'".format(string[:100]))


def checksum_file(path):
    path = os.path.expanduser(path)
    try:
        with open(path) as infile:
            return checksum(infile.read())
    except Exception:
        raise InternalException("failed to read '{}'".format(path))


def get_substring_idxs(substr, s):
    """
    Return a list of indexes of substr. If substr not found, list is
    empty.

    :param substr: Substring to match.
    :param s: String to match in.
    :param: List of integer substring start indices.
    """
    return [m.start() for m in re.finditer(substr, s)]


def package_path(path):
    abspath = resource_filename(__name__, path)
    if not os.path.exists(abspath):
        raise Data404Exception("package data '{}' does not exist"
                               .format(path))
    return abspath


def package_data(path):
    """
    Read package data file.

    :argument path: The relative path to the data file, e.g. 'share/foo.txt'.
    :return: File contents as byte string.
    :throws InternalException: in case of error.
    """
    package_path(path)

    try:
        return resource_string(__name__, path)
    except Exception:
        raise InternalException("failed to read package data '{}'"
                                .format(path))


def package_str(path):
    """
    Read package data file as a string.

    :argument path: The relative path to the text file, e.g. 'share/foo.txt'.
    :return: File contents as a string.
    :throws InternalException: in case of error.
    """
    try:
        return package_data(path).decode('utf-8')
    except UnicodeDecodeError:
        raise InternalException("failed to decode package data '{}'"
                                .format(path))


def sql_script(name):
    """
    Read SQL script to string.

    :argument name: The name of the SQL script (without file
                    extension), e.g. 'foo'.
    :return: SQL script as a string.
    :throws InternalException: in case of error.
    """
    path = os.path.join('share', 'sql', name + str('.sql'))
    return package_str(path)


def print_json(data):
    """
    Pretty print JSON.

    :param data: JSON blob.
    """
    print(json.dumps(data, sort_keys=True, indent=2, separators=(',', ': ')))


def read_csv(path, asdict=True):
    with open(path) as infile:
        if asdict:
            reader = csv.DictReader(infile)
        else:
            reader = csv.reader(infile)
        return [row for row in reader]


def minify_json(string, strip_space=True):
    """A port of the `JSON-minify` utility to the Python language.

    Based on JSON.minify.js: https://github.com/getify/JSON.minify

    Contributers:
    - Gerald Storer
     - Contributed original version
    - Felipe Machado
     - Performance optimization
    - Pradyun S. Gedam
     - Conditions and variable names changed
     - Reformatted tests and moved to separate file
     - Made into a PyPI Package
    """
    tokenizer = re.compile('"|(/\*)|(\*/)|(//)|\n|\r')
    end_slashes_re = re.compile(r'(\\)*$')

    in_string = False
    in_multi = False
    in_single = False

    new_str = []
    index = 0

    for match in re.finditer(tokenizer, string):
        if not (in_multi or in_single):
            tmp = string[index:match.start()]
            if not in_string and strip_space:
                # replace white space as defined in standard
                tmp = re.sub('[ \t\n\r]+', '', tmp)
            new_str.append(tmp)

        index = match.end()
        val = match.group()

        if val == '"' and not (in_multi or in_single):
            escaped = end_slashes_re.search(string, 0, match.start())

            # start of string or unescaped quote character to end string
            if not in_string or (escaped is None or len(escaped.group()) % 2 == 0):  # noqa
                in_string = not in_string
            index -= 1  # include " character in next catch
        elif not (in_string or in_multi or in_single):
            if val == '/*':
                in_multi = True
            elif val == '//':
                in_single = True
        elif val == '*/' and in_multi and not (in_string or in_single):
            in_multi = False
        elif val in '\r\n' and not (in_multi or in_string) and in_single:
            in_single = False
        elif not ((in_multi or in_single) or (val in ' \r\n\t' and strip_space)):  # noqa
            new_str.append(val)

    new_str.append(string[index:])
    return ''.join(new_str)
