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
import json
import os
import platform
import psutil
import re
import six
import sys
import tarfile

from collections import Mapping
from contextlib import contextmanager
from copy import deepcopy
from hashlib import sha1
from pkg_resources import resource_filename, resource_string, require

import labm8
from labm8 import fs
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


def checksum(data) -> str:
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


def checksum_list(*elems) -> str:
    """
    Checksum all elements of a list.

    Arguments:
        *elems: List of stringifiable data.

    Returns:
        str: Checksum.
    """
    string = "".join(sorted(str(x) for x in elems))
    return checksum_str(string)


def checksum_str(string: str) -> str:
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


def checksum_file(*path_components) -> str:
    """
    Checksum a file.

    Arguments:
        path_components (str): Path.

    Returns:
        str: Checksum.
    """
    path = must_exist(*path_components)

    try:
        with open(path, 'rb') as infile:
            return checksum(infile.read())
    except Exception:
        raise CLgenError("failed to read '{}'".format(path))


def unpack_archive(*components, **kwargs) -> str:
    """
    Unpack a compressed archive.

    Arguments:
        *components (str[]): Absolute path.
        **kwargs (dict, optional): Set "compression" to compression type.
            Default: bz2. Set "dir" to destination directory. Defaults to the
            directory of the archive.

    Returns:
        str: Path to directory.
    """
    path = fs.abspath(*components)
    compression = kwargs.get("compression", "bz2")
    dir = kwargs.get("dir", fs.dirname(path))

    fs.cd(dir)
    tar = tarfile.open(path, "r:" + compression)
    tar.extractall()
    tar.close()
    fs.cdpop()

    return dir


def update(dst: dict, src: dict) -> dict:
    """
    Recursively update values in dst from src.

    Unlike the builtin dict.update() function, this method will decend into
    nested dicts, updating all nested values.

    Arguments:
        dst (dict): Destination dict.
        src (dict): Source dict.

    Returns:
        dict: dst updated with entries from src.
    """
    for k, v in src.items():
        if isinstance(v, Mapping):
            r = update(dst.get(k, {}), v)
            dst[k] = r
        else:
            dst[k] = src[k]
    return dst


def dict_values(src: dict) -> list:
    """
    Recursively get values in dict.

    Unlike the builtin dict.values() function, this method will descend into
    nested dicts, returning all nested values.

    Arguments:
        src (dict): Source dict.

    Returns:
        list: List of values.
    """
    for v in src.values():
        if isinstance(v, dict):
            yield from dict_values(v)
        else:
            yield v


def get_substring_idxs(substr: str, s: str):
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


def format_json(data: dict) -> str:
    """
    Pretty print JSON.

    Arguments:
        data (dict): JSON blob.

    Returns:
        str: Formatted JSON
    """
    return json.dumps(data, sort_keys=True, indent=2, separators=(',', ': '))


def loads(text, **kwargs):
    """
    Deserialize `text` (a `str` or `unicode` instance containing a JSON
    document with Python or JavaScript like comments) to a Python object.

    Taken from `commentjson <https://github.com/vaidik/commentjson>`_, written
    by `Vaidik Kapoor <https://github.com/vaidik>`_.

    Copyright (c) 2014 Vaidik Kapoor, MIT license.

    :param text: serialized JSON string with or without comments.
    :param kwargs: all the arguments that `json.loads
                   <http://docs.python.org/2/library/json.html#json.loads>`_
                   accepts.
    :returns: `dict` or `list`.
    """
    regex = r'\s*(#|\/{2}).*$'
    regex_inline = r'(:?(?:\s)*([A-Za-z\d\.{}]*)|((?<=\").*\"),?)(?:\s)*(((#|(\/{2})).*)|)$'
    lines = text.split('\n')

    for index, line in enumerate(lines):
        if re.search(regex, line):
            if re.search(r'^' + regex, line, re.IGNORECASE):
                lines[index] = ""
            elif re.search(regex_inline, line):
                lines[index] = re.sub(regex_inline, r'\1', line)

    return json.loads('\n'.join(lines), **kwargs)


def load_json_file(path: str, must_exist: bool=True):
    """
    Load a JSON data blob.

    Arguments:
        path (str): Path to file.
        must_exist (bool, otional): If False, return empty dict if file does
            not exist.

    Returns:
        array or dict: JSON data.

    Raises:
        File404: If path does not exist, and must_exist is True.
        InvalidFile: If JSON is malformed.
    """
    try:
        with open(_must_exist(path)) as infile:
            return loads(infile.read())
    except ValueError as e:
        raise InvalidFile(
            "malformed JSON file '{path}'. Message from parser: {err}"
            .format(path=os.path.basename(path)), err=str(e))
    except File404 as e:
        if must_exist:
            raise e
        else:
            return {}


@contextmanager
def terminating(thing):
    """
    Context manager to terminate object at end of scope.
    """
    try:
        yield thing
    finally:
        thing.terminate()


def write_file(path: str, contents: str) -> None:
    if fs.dirname(path):
        fs.mkdir(fs.dirname(path))
    with open(path, 'w') as outfile:
        outfile.write(contents)


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


def main(model, sampler, print_corpus_dir=False, print_model_dir=False,
         print_sampler_dir=False) -> None:
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

    if model.endswith(".tar.bz2"):
        model = clgen.model.from_tar(model)
    else:
        model_json = load_json_file(model)
        model = clgen.model.from_json(model_json)

    sampler_json = load_json_file(sampler)
    sampler = clgen.sampler.from_json(sampler_json)

    # print cache paths
    if print_corpus_dir:
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
