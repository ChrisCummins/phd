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
OpenCL utilities
"""
from __future__ import absolute_import, division, print_function

import numpy as np
import re
import sys

import clgen


class OpenCLUtilException(clgen.CLgenError): pass
class PrototypeException(OpenCLUtilException): pass
class UnknownTypeException(PrototypeException): pass


class KernelArg(clgen.CLgenObject):
    """
    OpenCL Kernel Argument.

    Requires source code to have been pre-processed.
    """
    def __init__(self, string):
        self._string = string.strip()
        self._components = self._string.split()

        try:
            if "restrict" in self._components:
                self._is_restrict = True
                self._components.remove("restrict")
            else:
                self._is_restrict = False

            if "__restrict" in self._components:
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
        return self._string

    @property
    def components(self):
        return self._components

    @property
    def name(self):
        return self._components[-1]

    @property
    def type(self):
        return self._components[-2]

    @property
    def is_restrict(self):
        return self._is_restrict

    @property
    def qualifiers(self):
        return self._components[:-2]

    @property
    def is_pointer(self):
        return self.type[-1] == '*'

    @property
    def is_vector(self):
        idx = -2 if self.is_pointer else -1
        return self.type[idx].isdigit()

    @property
    def vector_width(self):
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

            KernelArg("float4*").bare_type == "float"
            KernelArg("uchar32").bare_type == "uchar"
        """
        try:
            return self._bare_type
        except AttributeError:  # set
            self._bare_type = re.sub(r'([0-9]+)?\*?$', '', self.type)
            return self._bare_type

    @property
    def is_const(self):
        try:
            return self._is_const
        except AttributeError:  # set
            self._is_const = True if 'const' in self.qualifiers else False
            return self._is_const

    @property
    def is_global(self):
        try:
            return self._is_global
        except AttributeError:  # set
            self._is_global = (True if
                               '__global' in self.qualifiers or
                               'global' in self.qualifiers
                               else False)
            return self._is_global

    @property
    def is_local(self):
        try:
            return self._is_local
        except AttributeError:  # set
            self._is_local = (True if
                              '__local' in self.qualifiers or
                              'local' in self.qualifiers
                              else False)
            return self._is_local

    @property
    def numpy_type(self):
        """
        Return the numpy data type associated with this argument.

        Return:

            numpy type

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
        self._string = ' '.join(string.split())
        if not self._string.startswith('__kernel void '):
            raise PrototypeException('malformed prototype', self._string)

    @property
    def name(self):
        try:
            return self._name
        except AttributeError:  # set
            idx_start = len('__kernel void ')
            idx_end = self._string.find('(')
            self._name = self._string[idx_start:idx_end]
            return self._name

    @property
    def args(self):
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
        return extract_prototype(src)


def get_attribute_range(s, start_idx):
    i = s.find('(', start_idx) + 1
    d = 1
    while i < len(s) and d > 0:
        if s[i] == '(':
            d += 1
        elif s[i] == ')':
            d -= 1
        i += 1

    return (start_idx, i)


def strip_attributes(src):
    idxs = sorted(clgen.get_substring_idxs('__attribute__', src))
    ranges = [get_attribute_range(src, i) for i in idxs]
    for r in reversed(ranges):
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
