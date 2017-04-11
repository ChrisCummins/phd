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

from pathlib import Path
from subprocess import Popen, PIPE
from tempfile import NamedTemporaryFile
from typing import List

import numpy as np

from pycparser import preprocess_file
from pycparser.c_ast import FileAST, NodeVisitor, PtrDecl, TypeDecl, Struct, IdentifierType
from pycparser.plyparser import ParseError
from pycparserext.ext_c_parser import OpenCLCParser


class OpenCLPreprocessError(ValueError):
    """
    Raised if pre-processor fails fails.

    Attributes
    ----------
    command : str
        Pre-processor invocation
    stdout : str
        Pre-processor output
    stderr : str
        Pre-processor error output
    """
    def __init__(self, command: str, stdout: str, stderr: str):
        super(OpenCLPreprocessError, self).__init__(command)
        self.command = command
        self.stdout = stdout
        self.stderr = stderr

    def __repr__(self) -> str:
        return self.command


class OpenCLValueError(ValueError):
    """ Raised if there is an invalid value OpenCL code """
    pass


class KernelArg(object):
    """
    OpenCL kernel argument representation.

    TODO
    ----
    * Attribute 'numpy_type' should depend on the properties of the device.
      E.g. not all devices will have 32 bit integer widths.
    """
    def __init__(self, ast):
        self.ast = ast

        # determine pointer type
        self.is_pointer = isinstance(self.ast.type, PtrDecl)
        self.address_space = "private"

        self.name = self.ast.name if self.ast.name else ""
        self.quals = self.ast.quals
        if len(self.quals):
            self.quals_str = " ".join(self.quals) + " "
        else:
            self.quals_str = ""

        # determine tyename
        try:
            if isinstance(self.ast.type.type, IdentifierType):
                typenames = self.ast.type.type.names
            elif isinstance(self.ast.type.type.type, Struct):
                typenames = ["struct", self.ast.type.type.type.name]
            else:
                typenames = self.ast.type.type.type.names
        except AttributeError as e:  # e.g. structs
            raise ValueError(
                f"unsupported data type for argument '{self.name}'") from e

        self.typename = " ".join(typenames)
        self.bare_type = self.typename.rstrip('0123456789')

        numpy_types = {
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
        }
        try:
            self.numpy_type = numpy_types[self.bare_type]
        except KeyError:
            supported_types_str = ",".join(sorted(numpy_types.keys()))
            raise OpenCLValueError(f"""\
unsupported type '{self.typename}' for argument \
'{self.quals_str}{self.typename} {self.name}'. \
supported types are: {{{supported_types_str}}}""")

        # get address space
        if self.is_pointer:
            address_quals = []
            if "local" in self.ast.quals:
                address_quals.append("local")

            if "__local" in self.ast.quals:
                address_quals.append("local")

            if "global" in self.ast.quals:
                address_quals.append("global")

            if "__global" in self.ast.quals:
                address_quals.append("global")

            if "constant" in self.ast.quals:
                address_quals.append("constant")

            if "__constant" in self.ast.quals:
                address_quals.append("constant")

            err_prefix = f"pointer argument '{self.quals_str}{self.typename} {self.name}'"
            if len(address_quals) == 1:
                self.address_space = address_quals[0]
            elif len(address_quals) > 1:
                raise OpenCLValueError(
                    f"{err_prefix} has multiple address space qualifiers")
            else:
                raise OpenCLValueError(
                    f"{err_prefix} has no address space qualifier")

        self.is_vector = self.typename[-1].isdigit()
        self.is_const = "const" in self.quals or self.address_space == "constant"

        if self.is_vector:
            m = re.search(r'([0-9]+)\*?$', self.typename)
            self.vector_width = int(m.group(1))
        else:
            self.vector_width = 1

    def __repr__(self):
        s = self.quals if len(self.quals) else []
        s.append(self.typename)
        if self.is_pointer:
            s.append("*")
        if self.name:
            s.append(self.name)
        return " ".join(s)


class ArgumentExtractor(NodeVisitor):
    """
    Extract kernel arguments from an OpenCL AST.

    Attributes
    ----------
    args : List[KernelArg]
        List of KernelArg instances.

    Raises
    ------
    ValueError
        If source contains more than one kernel definition.

    TODO
    ----
    * build a table of typedefs and substitute the original types when
      constructing kernel args.
    * handle structs by creating numpy types.
    """
    def __init__(self, *args, **kwargs):
        super(ArgumentExtractor, self).__init__(*args, **kwargs)
        self.kernel_count = 0
        self._args: List[KernelArg] = []

    def visit_FuncDef(self, node):
        # only visit kernels, not allfunctions
        if ("kernel" in node.decl.funcspec or
            "__kernel" in node.decl.funcspec):
            self.kernel_count += 1
        else:
            return

        # ensure we've only visited one kernel
        if self.kernel_count > 1:
            raise LookupError(
                "source contains more than one kernel definition")

        # function may not have arguments
        if node.decl.type.args:
            for param in node.decl.type.args.params:
                self._args.append(KernelArg(param))

    @property
    def args(self):
        if self.kernel_count != 1:
            raise LookupError("source contains no kernel definitions.")
        return self._args


__parser = OpenCLCParser()


def preprocess(src: str, include_dirs: List[Path]=[]) -> str:
    include_dirs = [Path(p).expanduser() for p in include_dirs]  # expand '~'
    command = ['cpp'] + [f"-I{p}" for p in include_dirs] + ['-xc', '-']

    try:
        process = Popen(command, stdin=PIPE, stdout=PIPE, stderr=PIPE,
                     universal_newlines=True)
        stdout, stderr = process.communicate(src)
        if process.returncode != 0:
            raise OpenCLPreprocessError(" ".join(command), stdout, stderr)

        return stdout
    except OSError as e:
        c = " ".join(command)
        raise RuntimeError(f"preprocess command {c} failed")


def parse(src: str) -> FileAST:
    """
    Parse OpenCL source code.

    Parameters
    ----------
    src : str
        OpenCL kernel source.

    Returns
    -------
    FileAST
        Parsed AST.

    Raises
    ------
    OpenCLValueError
        The source is not well formed, e.g. it contains a syntax error, or
        invalid types.
    """
    try:
        ast = __parser.parse(src)
        # strip preprocesor line objects and rebuild the AST.
        # See: https://github.com/inducer/pycparserext/issues/27
        children = [x[1] for x in ast.children() if not isinstance(x[1], list)]
        new_ast = FileAST(ext=children, coord=0)

        return new_ast
    except ParseError as e:
        raise OpenCLValueError("syntax error") from e


def extract_args(src: str) -> List[KernelArg]:
    """
    Extract kernel arguments for an OpenCL kernel.

    Accepts the source code for an OpenCL kernel and returns a list of its
    arguments.

    Parameters
    ----------
    src : str
        The OpenCL kernel source.

    Returns
    -------
    List[KernelArg]
        A list of the kernel's arguments, in order.

    Raises
    ------
    LookupError
        If the source contains no OpenCL kernel definitions, or more than one.
    ValueError
        If one of the kernel's parameter types are unsupported.

    Examples
    --------
    >>> args = extract_args("void kernel A(global float *a, const int b) {}")
    >>> args
    [global float * a, const int b]
    >>> args[0].typename
    'float'
    >>> args[0].address_space
    'global'
    >>> args[1].is_pointer
    False

    >>> extract_args("void kernel A() {}")
    []

    >>> extract_args("void /@&&& syn)")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    OpenCLValueError

    >>> extract_args("")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    LookupError

    >>> extract_args("void kernel A() {} void kernel B() {}")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    LookupError

    TODO
    ----
    * Pre-process source code.
    """
    visitor = ArgumentExtractor()
    visitor.visit(parse(src))
    return visitor.args
