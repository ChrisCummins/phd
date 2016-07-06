import os

from hashlib import sha1

from pkg_resources import resource_filename,resource_string

class SmithException(Exception): pass
class InternalException(SmithException): pass
class Data404Exception(InternalException): pass


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


def package_data(path):
    """
    Read package data file.

    :argument path: The relative path to the data file, e.g. 'share/foo.txt'.
    :return: File contents as byte string.
    :throws InternalException: in case of error.
    """
    abspath = resource_filename(__name__, path)
    if not os.path.exists(abspath):
        raise Data404Exception("package data '{}' does not exist"
                               .format(path))

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
