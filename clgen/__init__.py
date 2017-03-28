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
"""
import json
import os
import platform
import psutil
import re
import six
import sys
import tarfile

from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha1
from pkg_resources import resource_filename, resource_string, require

import labm8
from labm8 import cache
from labm8 import fs
from labm8 import jsonutil
from labm8 import system

from clgen import config as cfg


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

    *DO NOT* try to parse this or derive any special major/minor version
    information from it. Treat it as an opaque char array. The only valid
    operators for comparing versions are == and !=.

    Returns:
        str: Version string.
    """
    return require("clgen")[0].version


def cachepath(*relative_path_components: list) -> str:
    """
    Return path to file system cache.

    Arguments:
        *relative_path_components (list of str): Relative path of cache.

    Returns:
        str: Absolute path of file system cache.
    """
    assert(relative_path_components)

    cache_root = ["~", ".cache", "clgen", version()]
    return fs.path(*cache_root, *relative_path_components)


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
    if cfg.USE_CUDA:
        features_str = "(with CUDA)"
    elif cfg.USE_OPENCL:
        features_str = "(with OpenCL)"
    else:
        features_str = ""

    printfn("CLgen:     ", version(), features_str)
    printfn("Platform:  ", platform.system())
    printfn("Memory:    ",
            round(psutil.virtual_memory().total / (1024 ** 2)), "MB")

    if not cfg.USE_OPENCL:
        printfn()
        printfn("Device:     None")

    import pyopencl as cl
    for pltfm in cl.get_platforms():
        ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, pltfm)])
        for device in ctx.get_info(cl.context_info.DEVICES):
            devtype = cl.device_type.to_string(
                device.get_info(cl.device_info.TYPE))
            dev = device.get_info(cl.device_info.NAME)

            printfn()
            printfn("Device:    ", devtype, dev)
            printfn("Compute #.:", device.get_info(
                cl.device_info.MAX_COMPUTE_UNITS))
            printfn("Frequency: ", device.get_info(
                cl.device_info.MAX_CLOCK_FREQUENCY), "HZ")
            printfn("Memory:    ", round(
                device.get_info(
                    cl.device_info.GLOBAL_MEM_SIZE) / (1024 ** 2)), "MB")
            printfn("Driver:    ",
                    device.get_info(cl.device_info.DRIVER_VERSION))


def main(model, sampler, print_file_list=False, print_corpus_dir=False,
         print_model_dir=False, print_sampler_dir=False, quiet=False) -> None:
    """
    Main entry point for clgen.

    Arguments:
        model (str): Path to model.
        sample (str): Path to sampler.
        print_corpus_dir (bool, optional): If True, print cache path and exit.
        print_model_dir (bool, optional): If True, print cache path and exit.
        print_sampler_dir (bool, optional): If True, print cache path and exit.
    """
    import clgen.model
    import clgen.sampler
    from clgen import log

    model_json = jsonutil.read_file(model)
    model = clgen.model.from_json(model_json)

    sampler_json = jsonutil.read_file(sampler)
    sampler = clgen.sampler.from_json(sampler_json)

    # print cache paths
    if print_file_list:
        files = sorted(
            fs.ls(model.corpus.cache.path, abspaths=True, recursive=True) +
            fs.ls(model.cache.path, abspaths=True, recursive=True) +
            fs.ls(sampler.cache(model).path, abspaths=True, recursive=True))
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
    sampler.sample(model, quiet=quiet)
