import os

from pkg_resources import resource_filename,resource_string

class InternalException(Exception): pass

def package_data(path):
    """
    Read package data file.

    :argument path: The relative path to the data file, e.g. 'share/foo.txt'.
    :return: File contents as byte string.
    :throws InternalException: in case of error.
    """
    abspath = resource_filename(__name__, path)
    if not os.path.exists(abspath):
        raise InternalException("package data '{}' does not exist"
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
