# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
# This file is part of cldrive.
#
# cldrive is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cldrive is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cldrive.  If not, see <https://www.gnu.org/licenses/>.
"""OpenCL argument and type handling."""
import re
import typing

import numpy as np
from pycparser.c_ast import FileAST
from pycparser.c_ast import IdentifierType
from pycparser.c_ast import NodeVisitor
from pycparser.c_ast import PtrDecl
from pycparser.c_ast import Struct
from pycparser.plyparser import ParseError
from pycparserext.ext_c_parser import OpenCLCParser

# A lookup table mapping OpenCL data type names to the corresponding numpy data
# type.
NUMPY_TYPES = {
  "bool": np.dtype("bool"),
  "char": np.dtype("int8"),
  "double": np.dtype("float64"),
  "float": np.dtype("float32"),
  "half": np.dtype("uint8"),
  "int": np.dtype("int32"),
  "long": np.dtype("int64"),
  "short": np.dtype("int16"),
  "uchar": np.dtype("uint8"),
  "uint": np.dtype("uint32"),
  "ulong": np.dtype("uint64"),
  "unsigned": np.dtype("uint32"),
  "unsigned char": np.dtype("uint8"),
  "unsigned int": np.dtype("uint32"),
  "unsigned long": np.dtype("uint64"),
  "unsigned short": np.dtype("uint16"),
  "ushort": np.dtype("uint16"),
}

# The inverse lookup table of NUMPY_TYPES.
OPENCL_TYPES = dict((v, k) for k, v in NUMPY_TYPES.items())

# C printf() function format specifiers for numpy types.
FORMAT_SPECIFIERS = {
  np.dtype("bool"): "%d",
  np.dtype("float32"): "%.3f",
  np.dtype("float64"): "%.3f",
  np.dtype("int16"): "%hd",
  np.dtype("int32"): "%d",
  np.dtype("int64"): "%ld",
  np.dtype("int8"): "%hd",
  np.dtype("uint16"): "%hu",
  np.dtype("uint32"): "%u",
  np.dtype("uint64"): "%lu",
  np.dtype("uint8"): "%hd",
}

# Private OpenCL parser instance.
_OPENCL_PARSER = OpenCLCParser()


class OpenCLPreprocessError(ValueError):
  """Raised if pre-processor fails.

  Attributes:
    command: Pre-processor invocation.
    stdout: Pre-processor output.
    stderr: Pre-processor error output.
  """

  def __init__(self, command: str, stdout: str, stderr: str):
    super(OpenCLPreprocessError, self).__init__(command)
    self.command = command
    self.stdout = stdout
    self.stderr = stderr

  def __repr__(self) -> str:
    return self.command


class OpenCLValueError(ValueError):
  """Raised if there is an invalid OpenCL code."""

  pass


class MultipleKernelsError(LookupError):
  """Raised if source contains multiple kernels."""

  pass


class NoKernelError(LookupError):
  """Raised if a source does not contain a kernel."""

  pass


class KernelArg(object):
  """OpenCL kernel argument representation.

  TODO(cec): Attribute 'numpy_type' should depend on the properties of the
    device. E.g. not all devices will have 32 bit integer widths.
  """

  def __init__(self, ast):
    self.ast = ast

    # Determine pointer type.
    self.is_pointer = isinstance(self.ast.type, PtrDecl)
    self.address_space = "private"

    self.name = self.ast.name if self.ast.name else ""
    self.quals = self.ast.quals
    if len(self.quals):
      self.quals_str = " ".join(self.quals) + " "
    else:
      self.quals_str = ""

    # Determine type name.
    try:
      if isinstance(self.ast.type.type, IdentifierType):
        type_names = self.ast.type.type.names
      elif isinstance(self.ast.type.type.type, Struct):
        type_names = ["struct", self.ast.type.type.type.name]
      else:
        type_names = self.ast.type.type.type.names
    except AttributeError as e:  # e.g. structs
      raise ValueError(
        f"Unsupported data type for argument: '{self.name}'"
      ) from e

    self.typename = " ".join(type_names)
    self.bare_type = self.typename.rstrip("0123456789")

    # Get address space.
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

      err_prefix = (
        "Pointer argument " f"'{self.quals_str}{self.typename} *{self.name}'"
      )
      if len(address_quals) == 1:
        self.address_space = address_quals[0]
      elif len(address_quals) > 1:
        raise OpenCLValueError(
          f"{err_prefix} has multiple address space qualifiers"
        )
      else:
        raise OpenCLValueError(f"{err_prefix} has no address space qualifier")

    self.is_vector = self.typename[-1].isdigit()
    self.is_const = "const" in self.quals or self.address_space == "constant"

    if self.is_vector:
      m = re.search(r"([0-9]+)\*?$", self.typename)
      self.vector_width = int(m.group(1))
    else:
      self.vector_width = 1

  @property
  def numpy_type(self):
    """Get the numpy equivalent for the argument."""
    try:
      return NUMPY_TYPES[self.bare_type]
    except KeyError:
      supported_types_str = ",".join(sorted(NUMPY_TYPES.keys()))
      raise OpenCLValueError(
        f"""\
Unsupported type '{self.typename}' for argument \
'{self.quals_str}{self.typename} {self.name}'. \
Supported types are: {{{supported_types_str}}}"""
      )

  def __repr__(self):
    s = self.quals if len(self.quals) else []
    s.append(self.typename)
    if self.is_pointer:
      s.append("*")
    if self.name:
      s.append(self.name)
    return " ".join(s)


class ArgumentExtractor(NodeVisitor):
  """Extract kernel arguments from an OpenCL AST.

  TODO(cec): Build a table of typedefs and substitute the original types when
    constructing kernel args.
  TODO(cec): Handle structs by creating numpy types.

  Attributes:
    args: typing.List of KernelArg instances.
    name: Kernel name.

  Raises:
    ValueError: If source contains more than one kernel definition.
  """

  def __init__(self, *args, **kwargs):
    self.extract_args = kwargs.pop("extract_args", True)

    super(ArgumentExtractor, self).__init__(*args, **kwargs)
    self.kernel_count = 0
    self._args: typing.List[KernelArg] = []
    self.name = None

  def visit_FuncDef(self, node):
    # Only visit kernels, not all functions.
    if "kernel" in node.decl.funcspec or "__kernel" in node.decl.funcspec:
      self.kernel_count += 1
      self.name = node.decl.name
    else:
      return

    # Ensure we've only visited one kernel.
    if self.kernel_count > 1:
      raise MultipleKernelsError(
        "Source contains more than one kernel definition"
      )

    # Function may not have arguments
    if self.extract_args and node.decl.type.args:
      for param in node.decl.type.args.params:
        self._args.append(KernelArg(param))

  @property
  def args(self) -> typing.List[KernelArg]:
    """Get the kernels for the kernel."""
    if self.kernel_count != 1:
      raise NoKernelError("Source contains no kernel definitions")
    return self._args


def ParseSource(src: str) -> FileAST:
  """Parse OpenCL source code.

  Args:
    src: OpenCL kernel source.

  Returns:
    The parsed AST.

  Raises:
    OpenCLValueError: If the source is not well formed, e.g. it contains a
      syntax error, or invalid types.
  """
  try:
    ast = _OPENCL_PARSER.parse(src)
    # Strip pre-procesor line objects and rebuild the AST.
    # See: https://github.com/inducer/pycparserext/issues/27
    children = [x[1] for x in ast.children() if not isinstance(x[1], list)]
    new_ast = FileAST(ext=children, coord=0)
    return new_ast
  except (ParseError, AssertionError) as e:
    raise OpenCLValueError(f"Syntax error: '{e}'") from e


def GetKernelArguments(src: str) -> typing.List[KernelArg]:
  """Extract arguments for an OpenCL kernel.

  Accepts the source code for an OpenCL kernel and returns a list of its
  arguments.

  TODO(cec): Pre-process the source code.

  Args:
    src: The OpenCL kernel source.

  Returns:
    A list of the kernel's arguments, in order.

  Raises:
    LookupError: If the source contains no OpenCL kernel definitions, or more
      than one.
    ValueError: If one of the kernel's parameter types are unsupported.

  Examples:
    >>> args = GetKernelArguments("void kernel A(global float *a, const int b) {}")
    >>> args
    [global float * a, const int b]
    >>> args[0].typename
    'float'
    >>> args[0].address_space
    'global'
    >>> args[1].is_pointer
    False

    >>> GetKernelArguments("void kernel A() {}")
    []

    >>> GetKernelArguments("void /@&&& syn)")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    OpenCLValueError

    >>> GetKernelArguments("")  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ...
    NoKernelError
  """
  visitor = ArgumentExtractor()
  visitor.visit(ParseSource(src))
  return visitor.args


def GetKernelName(src: str) -> str:
  """Extract the name of an OpenCL kernel.

  Accepts the source code for an OpenCL kernel and returns its name.

  Args:
    src: The OpenCL kernel source.

  Returns:
    str: The kernel name.

  Raises:
    NoKernelError: If the source contains no OpenCL kernel definitions.
    MultipleKernelsError: If the source contains multiple OpenCL kernel
      definitions.

  Examples:
    >>> GetKernelName("void kernel foo() {}")
    'foo'
    >>> GetKernelName("void kernel A(global float *a, const int b) {}")
    'A'
  """
  visitor = ArgumentExtractor(extract_args=False)
  visitor.visit(ParseSource(src))
  if visitor.name:
    return visitor.name
  else:
    raise NoKernelError("Source contains no kernel definitions")
