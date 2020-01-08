# This file is part of libcecl.
#
# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# libcecl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# libcecl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with libcecl.  If not, see <https://www.gnu.org/licenses/>.
"""Compilation utilities for libcecl."""
import functools
import typing

from labm8.py import app
from labm8.py import bazelutil
from labm8.py import system

FLAGS = app.FLAGS

OPENCL_HEADERS_DIR = bazelutil.DataPath("opencl_120_headers")

# Path to OpenCL headers and library.
if system.is_linux():
  LIBOPENCL_DIR = bazelutil.DataPath("libopencl")

LIBCECL_SO = bazelutil.DataPath("phd/gpu/libcecl/libcecl.so")
LIBCECL_HEADER = bazelutil.DataPath("phd/gpu/libcecl/libcecl.h")


def _OpenClCompileAndLinkFlags() -> typing.Tuple[
  typing.List[str], typing.List[str]
]:
  """Private helper method to get device-specific OpenCL flags."""
  if system.is_linux():
    return (
      ["-isystem", str(OPENCL_HEADERS_DIR)],
      [
        f"-L{LIBOPENCL_DIR}",
        f"-Wl,-rpath,{LIBOPENCL_DIR}",
        "-lOpenCL",
        "-DCL_SILENCE_DEPRECATION",
      ],
    )
  else:  # macOS
    return (
      ["-isystem", str(OPENCL_HEADERS_DIR)],
      ["-framework", "OpenCL", "-DCL_SILENCE_DEPRECATION"],
    )


@functools.lru_cache(maxsize=2)
def OpenClCompileAndLinkFlags(
  opencl_headers: bool = True,
) -> typing.Tuple[typing.List[str], typing.List[str]]:
  """Get device-specific OpenCL compile and link flags."""
  cflags, ldflags = _OpenClCompileAndLinkFlags()
  if not opencl_headers:
    cflags = []
  return cflags, ldflags


@functools.lru_cache(maxsize=2)
def LibCeclCompileAndLinkFlags(
  opencl_headers: bool = True,
) -> typing.Tuple[typing.List[str], typing.List[str]]:
  """Get device-specific LibCecl compile and link flags.

  WARNING: Executable compiled with these flags must be executed with the
  environment variables from LibCeclExecutableEnvironmentVariables() to set a
  correct LD_LIBRARY_PATH!
  """
  cflags, ldflags = OpenClCompileAndLinkFlags(opencl_headers=opencl_headers)
  return (
    cflags + ["-isystem", str(LIBCECL_HEADER.parent)],
    ldflags + ["-lcecl", f"-L{LIBCECL_SO.parent}"],
  )


def LibCeclExecutableEnvironmentVariables() -> typing.Dict[str, str]:
  """Return the environment variables which must be used when executing libcecl
  binaries.
  """
  return {
    "LD_LIBRARY_PATH": str(LIBCECL_SO.parent),
    "DYLD_LIBRARY_PATH": str(LIBCECL_SO.parent),
  }
