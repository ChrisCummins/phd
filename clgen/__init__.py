#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
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

Attributes
----------

__version__ : str
    PEP 440 compliant version string.

version_info : namedtuple['major', 'minor', 'micro', 'releaselevel'])
    Version tuple.
"""
import json
import labm8
import os
import platform
import psutil
import re
import six
import sys
import tarfile

from collections import namedtuple
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha1
from labm8 import cache
from labm8 import fs
from labm8 import jsonutil
from labm8 import system
from labm8 import system
from pkg_resources import resource_filename, resource_string, require

from clgen._config import *


__author__ = "Chris Cummins"
__copyright__ = "Copyright 2017, Chris Cummins"
__license__ = "GPL v3"
__version__ = require("clgen")[0].version
__maintainer__ = __author__
__email__ = "chrisc.101@gmail.com"
__status__ = "Prototype"


# version_info tuple
_major = int(__version__.split(".")[0])
_minor = int(__version__.split('.')[1]) if len(__version__.split('.')) > 1 else 0
_micro = int(__version__.split('.')[2]) if len(__version__.split('.')) > 2 else 0
_releaselevel = __version__.split('.')[3] if len(__version__.split('.')) > 3 else 'final'

version_info_t = namedtuple('version_info_t', ['major', 'minor', 'micro', 'releaselevel'])
version_info = version_info_t(_major, _minor, _micro, _releaselevel)


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


class UserError(CLgenError):
    """
    Raised in case of bad user interaction, e.g. an invalid argument.
    """
    pass


class File404(InternalError):
    """
    Data not found.
    """
    pass


class InvalidFile(UserError):
    """
    Raised in case a file contains invalid contents.
    """
    pass


class CLgenObject(object):
    """
    Base object for CLgen classes.
    """
    pass


def version() -> str:
    """
    Get the package version.

    Returns:
        str: Version string.
    """
    return __version__


def cachepath(*relative_path_components: list) -> str:
    """
    Return path to file system cache.

    Arguments:
        *relative_path_components (list of str): Relative path of cache.

    Returns:
        str: Absolute path of file system cache.
    """
    cache_root = ["~", ".cache", "clgen",
                  f"{version_info.major}.{version_info.minor}.x" ]
    fs.mkdir(*cache_root)
    return fs.path(*cache_root, *relative_path_components)


def get_default_author() -> str:
    """
    Get a default author name.

    If CLGEN_AUTHOR environment variable is set, use that. Else, author
    is $USER@$HOSTNAME.

    Returns:
        str: Author name.
    """
    return os.environ.get(
        "CLGEN_AUTHOR",
        "{user}@{host}".format(user=system.USERNAME, host=system.HOSTNAME))


def mkcache(*relative_path_components: list) -> cache.FSCache:
    """
    Instantiae a file system cache.

    If the cache does not exist, one is created.

    Arguments:
        *relative_path_components (list of str): Relative path of cache.

    Returns:
        labm8.FSCache: Filesystem cache.
    """

    return cache.FSCache(cachepath(*relative_path_components),
                         escape_key=cache.escape_path)


def must_exist(*path_components, **kwargs) -> str:
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

_must_exist = must_exist  # prevent variable scope shadowing


def package_path(*path) -> str:
    """
    Path to package file.

    Arguments:

        *path (str[]): Path components.

    Returns:

        str: Path.
    """
    path = os.path.expanduser(os.path.join(*path))
    abspath = resource_filename(__name__, path)
    return must_exist(abspath)


def data_path(*path) -> str:
    """
    Path to package file.

    Arguments:

        *path (str[]): Path components.

    Returns:

        str: Path.
    """
    return package_path("data", *path)


def package_data(*path) -> bytes:
    """
    Read package data file.

    Arguments:
        path (str): The relative path to the data file, e.g. 'share/foo.txt'.

    Returns:
        bytes: File contents.

    Raises:
        InternalError: In case of IO error.
    """
    # throw exception if file doesn't exist
    package_path(*path)

    try:
        return resource_string(__name__, fs.path(*path))
    except Exception:
        raise InternalError("failed to read package data '{}'".format(path))


def package_str(*path) -> str:
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


def sql_script(name: str) -> str:
    """
    Read SQL script to string.

    Arguments:
        name (str): The name of the SQL script (without file extension).

    Returns:
        str: SQL script.
    """
    path = fs.path('data', 'sql', str(name) + ".sql")
    return package_str(path)


@contextmanager
def terminating(thing):
    """
    Context manager to terminate object at end of scope.
    """
    try:
        yield thing
    finally:
        thing.terminate()


def platform_info(printfn=print) -> None:
    """
    Log platform information.

    Arguments:
        printfn (fn, optional): Function to call to print output to. Default
            `print()`.
    """
    if USE_CUDA:
        features_str = "(with CUDA)"
    else:
        features_str = ""

    printfn("CLgen:     ", version(), features_str)
    printfn("Platform:  ", platform.system())
    printfn("Memory:    ",
            round(psutil.virtual_memory().total / (1024 ** 2)), "MB")


def main(model: str, sampler: str, print_file_list: bool=False,
         print_corpus_dir: bool=False, print_model_dir: bool=False,
         print_sampler_dir: bool=False) -> None:
    """
    Main entry point for clgen.

    Arguments:
        model (str): Path to model.
        sample (str): Path to sampler.
        print_corpus_dir (bool, optional): If True, print cache path and exit.
        print_model_dir (bool, optional): If True, print cache path and exit.
        print_sampler_dir (bool, optional): If True, print cache path and exit.
    """
    from clgen import log

    model_json = jsonutil.read_file(model)
    model = clgen.Model.from_json(model_json)

    sampler_json = jsonutil.read_file(sampler)
    sampler = clgen.Sampler.from_json(sampler_json)

    # print cache paths
    if print_file_list:
        files = sorted(
            model.corpus.cache.ls(abspaths=True, recursive=True) +
            model.cache.ls(abspaths=True, recursive=True) +
            sampler.cache(model).ls(abspaths=True, recursive=True))
        print('\n'.join(files))
        sys.exit(0)
    elif print_corpus_dir:
        print(model.corpus.cache.path)
        sys.exit(0)
    elif print_model_dir:
        print(model.cache.path)
        sys.exit(0)
    elif print_sampler_dir:
        print(sampler.cache(model).path)
        sys.exit(0)

    model.train()
    sampler.sample(model)


# package level imports
from clgen._fetch import *
from clgen._explore import *
from clgen._atomizer import *
from clgen._corpus import *
from clgen._model import *
from clgen._preprocess import *
from clgen._sampler import *
