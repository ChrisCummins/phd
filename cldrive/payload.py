# Copyright (C) 2017 Chris Cummins.
#
# This file is part of cldrive.
#
# Cldrive is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Cldrive is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <http://www.gnu.org/licenses/>.
#
import re

from typing import List

import numpy as np

from pycparser.c_ast import NodeVisitor, PtrDecl, TypeDecl, IdentifierType
from pycparserext.ext_c_parser import OpenCLCParser

import cldrive


class KernelArg(object):
    """
    TODO:
        * Attribute 'numpy_type' should depend on the properties of the device.
          E.g. not all devices will have 32 bit integer widths.
    """
    def __init__(self, typename: str, name="", quals: List[str]=[]):
        self.typename = typename
        self.name = name
        self.quals = quals

        self.bare_type = typename.rstrip('0123456789')

        self.is_vector = self.typename[-1].isdigit()

        self.is_const = "const" in self.quals

        if self.is_vector:
            m = re.search(r'([0-9]+)\*?$', self.typename)
            self.vector_width = int(m.group(1))
        else:
            self.vector_width = 1

        self.numpy_type = {
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
        if self.numpy_type is None:
            raise cldrive.KernelArgError(self.bare_type)

    @staticmethod
    def from_ast(argtype, node):
        tmp = node.type.type

        if isinstance(tmp, IdentifierType):
            typenames = tmp.names
        else:
            try:
                typenames = tmp.type.names
            except AttributeError:
                # e.g. structs
                raise cldrive.KernelArgError(node)

        if len(typenames) != 1:
            raise cldrive.KernelArgError(node)

        return argtype(typename=typenames[0], quals=node.quals, name=node.name)

    def __repr__(self):
        classname = self.__class__.__name__
        if len(self.quals):
            quals_str = " ".join(self.quals) + " "
        else:
            quals_str = ""
        return f"classname {quals_str}{self.typename} {self.name}"


class ScalarArg(KernelArg):
    is_pointer = False
    has_host_input = True


class LocalBufferArg(KernelArg):
    is_pointer = True
    has_host_input = False


class GlobalBufferArg(KernelArg):
    is_pointer = True
    has_host_input = True


class ArgumentExtractor(NodeVisitor):
    """
    Extract kernel arguments from an OpenCL AST.

    Attributes:
        args (List[KernelArg]): List of KernelArg instances.

    TODO:
        * build a table of typedefs and substitute the original types when
          constructing kernel args.
    """
    def __init__(self, *args, **kwargs):
        super(ArgumentExtractor, self).__init__(*args, **kwargs)
        self.kernel_count = 0
        self._args: List[KernelArg] = []

    def add_kernel_arg(self, param):
        """
        Add a kernel argument for the given param.

        Raises:
            ParseError: If the argument cannot be processed.
        """
        qualifiers = param.quals
        if isinstance(param.type, PtrDecl):
            if "local" in param.quals:
                param.quals.remove("local")
                argtype = LocalBufferArg
            elif "__local" in param.quals:
                param.quals.remove("__local")
                argtype = LocalBufferArg
            elif "global" in param.quals:
                param.quals.remove("global")
                argtype = GlobalBufferArg
            elif "__global" in param.quals:
                param.quals.remove("__global")
                argtype = GlobalBufferArg
            else:
                # the pointer argument is neither a global or float
                raise cldrive.ParseError(f"Argument '{param.name}' is neither "
                                         "global or float qualified")
        elif isinstance(param.type, TypeDecl):
            argtype = ScalarArg
        else:
            raise cldrive.ParseError(param)

        self._args.append(KernelArg.from_ast(argtype, param))

    @property
    def args(self):
        if self.kernel_count != 1:
            raise cldrive.ParseError("source contains no kernel definitions.")
        return self._args


    def visit_FuncDef(self, node):
        """
        Raises:
            ParseError: In case of a problem.
        """
        # only visit kernels, not allfunctions
        if ("kernel" in node.decl.funcspec or
            "__kernel" in node.decl.funcspec):
            self.kernel_count += 1
        else:
            return

        # ensure we've only visited one kernel
        if self.kernel_count > 1:
            raise cldrive.ParseError(
                "source contains more than one kernel definition")

        # function may not have arguments
        if node.decl.type.args:
            for param in node.decl.type.args.params:
                self.add_kernel_arg(param)

_parser = OpenCLCParser()

def parse(src: str):
    """
    TODO: Pre-process source.
    """
    return _parser.parse(src)