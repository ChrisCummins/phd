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
OpenCL utilities.
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import re

from six import string_types

import clgen
from clgen import config as cfg
from clgen import log


class OpenCLUtilException(clgen.CLgenError):
    """
    Module error.
    """
    pass


class PrototypeException(OpenCLUtilException):
    """
    Kernel prototype error.
    """
    pass


class UnknownTypeException(PrototypeException):
    """
    Bad or unsupported type.
    """
    pass


class KernelArg(clgen.CLgenObject):
    """
    OpenCL Kernel Argument.

    *Note:* Requires source code to have been pre-processed.
    """
    def __init__(self, string):
        """
        Create a kernel argument from a string.

        Arguments:
            string (str): OpenCL argument string.
        """
        assert(isinstance(string, string_types))

        self._string = string.strip()
        self._components = self._string.split()

        try:
            if "restrict" in self._components:
                self._is_restrict = True
                self._components.remove("restrict")
            elif "__restrict" in self._components:
                self._is_restrict = True
                self._components.remove("__restrict")
            else:
                self._is_restrict = False

            if "unsigned" in self._components:
                self._components.remove("unsigned")
                self._components[-2] = "unsigned " + self._components[-2]
        except Exception as e:
            raise PrototypeException(e)


    @property
    def string(self):
        """
        Return the original string.

        Examples:

            >>> KernelArg("__global float4* a").string
            "__global float4* a"
            >>> KernelArg("const int b").string
            "const int b"

        Returns:
            str: String, as passed to constructor.
        """
        return self._string

    @property
    def components(self):
        """
        Kernel argument components.

        Examples:

            >>> KernelArg("__global float4* a").components
            ["__global", "float4*", "a"]
            >>> KernelArg("const int b").components
            ["const", "int", "b"]

        Returns:
            str[]: Argument components.
        """
        return self._components

    @property
    def name(self):
        """
        Get the argument variable name.

        Examples:

            >>> KernelArg("__global float4* a").name
            "a"
            >>> KernelArg("const int b").name
            "b"

        Returns:
            str: Argument name.
        """
        return self._components[-1]

    @property
    def type(self):
        """
        Get the argument type.

        Examples:

            >>> KernelArg("__global float4* a").type
            "float4*"
            >>> KernelArg("const int b").type
            "int"

        Returns:
            str: Argument type, including pointer '*' symbol, if present.
        """
        return self._components[-2]

    @property
    def is_restrict(self):
        """
        Argument has restrict keyword.

        Examples:

            >>> KernelArg("__global float4* a").is_restrict
            False
            >>> KernelArg("restrict int* b").is_restrict
            True

        Returns:
            bool: True if restrict.
        """
        return self._is_restrict

    @property
    def qualifiers(self):
        """
        Return all argument type qualifiers.

        Examples:

            >>> KernelArg("__global float4* a").qualifiers
            ["__global"]
            >>> KernelArg("const int b").qualifiers
            ["const"]

        Returns:
            str[]: Type qualifiers.
        """
        return self._components[:-2]

    @property
    def is_pointer(self):
        """
        Returns whether argument is a pointer.

        Examples:

            >>> KernelArg("__global float4* a").is_pointer
            True
            >>> KernelArg("const int b").is_pointer
            False

        Returns:
            bool: True if pointer, else False.
        """
        return self.type[-1] == '*'

    @property
    def is_vector(self):
        """
        Returns whether argument is a vectory type, e.g. 'int4'.

        Examples:

            >>> KernelArg("__global float4* a").is_vector
            True
            >>> KernelArg("const int b").is_vector
            False

        Returns:
            bool: True if vector type, else False.
        """
        idx = -2 if self.is_pointer else -1
        return self.type[idx].isdigit()

    @property
    def vector_width(self):
        """
        Returns width of vector type.

        Examples:

            >>> KernelArg("__global float4* a").vector_width
            4
            >>> KernelArg("__global int* a").vector_width
            1

        Returns:
            int: Vector width.
        """
        try:
            return self._vector_width
        except AttributeError:  # set
            if self.is_vector:
                m = re.search(r'([0-9]+)\*?$', self.type)
                self._vector_width = int(m.group(1))
            else:
                self._vector_width = 1
            return self._vector_width

    @property
    def bare_type(self):
        """
        Type name, without vector or pointer qualifiers.

        Examples:

            >>> KernelArg("__global float4* a").bare_type
            "float"
            >>> KernelArg("const int b").bare_type
            "uchar"

        Returns:
            str: Bare type.
        """
        try:
            return self._bare_type
        except AttributeError:  # set
            self._bare_type = re.sub(r'([0-9]+)?\*?$', '', self.type)
            return self._bare_type

    @property
    def is_const(self):
        """
        Kernel arg is constant.

        Examples:

            >>> KernelArg("__global float4* a").is_const
            False
            >>> KernelArg("const int b").is_const
            True

        Returns:
            bool: True if const, else False.
        """
        try:
            return self._is_const
        except AttributeError:  # set
            self._is_const = True if 'const' in self.qualifiers else False
            return self._is_const

    @property
    def is_global(self):
        """
        Kernel arg is global.

        Examples:

            >>> KernelArg("__global float4* a").is_global
            True
            >>> KernelArg("const int b").is_global
            False

        Returns:
            bool: True if global, else False.
        """
        try:
            return self._is_global
        except AttributeError:  # set
            self._is_global = (
                True if '__global' in self.qualifiers or
                'global' in self.qualifiers else False)
            return self._is_global

    @property
    def is_local(self):
        """
        Kernel arg is local.

        Examples:

            >>> KernelArg("__local float4* a").is_local
            True
            >>> KernelArg("const int b").is_local
            False

        Returns:
            bool: True if local, else False.
        """
        try:
            return self._is_local
        except AttributeError:  # set
            self._is_local = (
                True if '__local' in self.qualifiers or
                'local' in self.qualifiers else False)
            return self._is_local

    @property
    def numpy_type(self):
        """
        Return the numpy data type associated with this argument.

        Examples:

            >>> KernelArg("__local float4* a").numpy_type
            np.float32
            >>> KernelArg("const int b").numpy_type
            np.int32

        Return:

            numpy type: Kernel type.

        Raises:

            UnknownTypeException: If type can't be deduced.
        """
        np_type = {
            "bool": np.bool_,
            "char": np.int8,
            "double": np.float64,
            "float": np.float32,
            "half": np.uint8,
            "int": np.int32,
            "long": np.int64,
            "short": np.int16,
            "uchar": np.uint8,
            "uint": np.uint32,
            "ulong": np.uint64,
            "unsigned char": np.uint8,
            "unsigned int": np.uint32,
            "unsigned long": np.uint64,
            "unsigned short": np.uint16,
            "ushort": np.uint16,
            "void": np.int64,
        }.get(self.bare_type, None)
        if np_type is None:
            raise UnknownTypeException(self.type)
        return np_type

    def __repr__(self):
        return self._string


class KernelPrototype(clgen.CLgenObject):
    """
    OpenCL Kernel Prototype.

    Requires source code to have been pre-processed.
    """
    def __init__(self, string):
        """
        Create OpenCL kernel prototype.

        *Note:* requires source to have been preprocessed.

        Arguments:
            string (str): Prototype string.
        """
        self._string = ' '.join(string.split())
        if not self._string.startswith('__kernel void '):
            raise PrototypeException('malformed prototype', self._string)

    @property
    def name(self):
        """
        Kernel function name.

        Examples:

            >>> KernelPrototype("__kernel void A() {").name
            "A"

        Returns:
            str: Kernel name.
        """
        try:
            return self._name
        except AttributeError:  # set
            idx_start = len('__kernel void ')
            idx_end = self._string.find('(')
            self._name = self._string[idx_start:idx_end]
            return self._name

    @property
    def args(self):
        """
        Kernel arguments.

        Returns:
            KernelArg[]: Kernel arguments.
        """
        try:
            return self._args
        except AttributeError:
            idx_open_brace = self._string.find('(')
            idx_close_brace = self._string.find(')')
            inner_brace = self._string[idx_open_brace + 1:idx_close_brace].strip()
            # Add special case for prototypes which have no args:
            if not inner_brace or inner_brace == 'void':
                self._args = []
            else:
                self._args = [KernelArg(x) for x in inner_brace.split(',')]
            return self._args

    def __repr__(self):
        return self._string

    @staticmethod
    def from_source(src):
        """
        Create KernelPrototype from OpenCL kernel source.

        Argument:
            src (str): OpenCL kernel source.

        Returns:
            KernelPrototype: Kernel prototype.
        """
        return extract_prototype(src)


def get_attribute_range(src, start_idx):
    """
    Get string indices range of attributes.

    Arguments:
        src (str): OpenCL kernel source.
        start_idx (int): Index of attribute opening brace.

    Returns:
        tuple of int: Start and end indices of attributes.
    """
    i = src.find('(', start_idx) + 1
    d = 1
    while i < len(src) and d > 0:
        if src[i] == '(':
            d += 1
        elif src[i] == ')':
            d -= 1
        i += 1

    return (start_idx, i)


def strip_attributes(src):
    """
    Remove attributes from OpenCL source.

    Arguments:
        src (str): OpenCL source.

    Returns:
        str: OpenCL source, with ((attributes)) removed.
    """
    # get list of __attribute__ substrings
    idxs = sorted(clgen.get_substring_idxs('__attribute__', src))

    # get ((attribute)) ranges
    attribute_ranges = [get_attribute_range(src, i) for i in idxs]

    # remove ((atribute)) ranges
    for r in reversed(attribute_ranges):
        src = src[:r[0]] + src[r[1]:]
    return src


def get_cl_kernel_end_idx(src, start_idx=0, max_len=5000):
    """
    Return the index of the character after the end of the OpenCL
    kernel.

    Arguments:

        src (str): OpenCL source.
        start_idx (int, optional): Start index.
        max_len (int, optional): Maximum kernel length.

    Returns:

        int: Index of end of OpenCL kernel.
    """
    i = src.find('{', start_idx) + 1
    d = 1  # depth
    while i < min(len(src), start_idx + max_len) and d > 0:
        if src[i] == '{':
            d += 1
        elif src[i] == '}':
            d -= 1
        i += 1
    return i


def get_cl_kernel(src, start_idx, max_len=5000):
    """
    Return the OpenCL kernel.

    Arguments:

        src (str): OpenCL source.
        start_idx (int, optional): Start index.
        max_len (int, optional): Maximum kernel length.

    Returns:

        str: OpenCL kernel.
    """
    return src[start_idx:get_cl_kernel_end_idx(src, start_idx)]


def get_cl_kernels(src):
    """
    Return OpenCL kernels.

    Arguments:

        src (str): OpenCL source.

    Returns:

        str[]: OpenCL kernels.
    """
    idxs = clgen.get_substring_idxs('__kernel', src)
    kernels = [get_cl_kernel(src, i) for i in idxs]
    return kernels


def extract_prototype(src):
    """
    Extract OpenCL kernel prototype from preprocessed file.

    Arguments:

        src (str): OpenCL source.

    Returns:

        KernelPrototype: Prototype instance.
    """
    idxs = clgen.get_substring_idxs('__kernel void ', src)
    if len(idxs) != 1:
        raise PrototypeException("Invalid number of kernels found: {}"
                                 .format(len(idxs)))
    src = get_cl_kernel(src, idxs[0])

    try:
        index = src.index('{') + 1
        prototype = src[:index]
    except ValueError:
        raise PrototypeException("malformed seed")

    return KernelPrototype(prototype)


def get_contexts_and_devices():
    """
    Instantiate OpenCL contexts for all platforms and return devices.

    Returns:
        dict: pyopencl.Context as key, pyopencl.Device[] as value
    """
    import pyopencl as cl

    results = {}
    for pltfm in cl.get_platforms():
        ctx = cl.Context(properties=[(cl.context_properties.PLATFORM, pltfm)])
        devices = ctx.get_info(cl.context_info.DEVICES)
        results[ctx] = devices

    return results
