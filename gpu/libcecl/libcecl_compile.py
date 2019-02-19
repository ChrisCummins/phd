"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
import functools
import typing

from absl import flags

from labm8 import bazelutil, system


FLAGS = flags.FLAGS

OPENCL_HEADERS_DIR = bazelutil.DataPath('opencl_120_headers')

# Path to OpenCL headers and library.
if system.is_linux():
  LIBOPENCL_DIR = bazelutil.DataPath('libopencl')

LIBCECL_SO = bazelutil.DataPath('phd/gpu/libcecl/libcecl.so')
LIBCECL_HEADER = bazelutil.DataPath('phd/gpu/libcecl/libcecl.h')


def _OpenClCompileAndLinkFlags(
) -> typing.Tuple[typing.List[str], typing.List[str]]:
  """Private helper method to get device-specific OpenCL flags."""
  if system.is_linux():
    return (['-isystem', str(OPENCL_HEADERS_DIR)],
            [f'-L{LIBOPENCL_DIR}', f'-Wl,-rpath,{LIBOPENCL_DIR}', '-lOpenCL'])
  else:  # macOS
    return ['-isystem', str(OPENCL_HEADERS_DIR)], ['-framework', 'OpenCL']


@functools.lru_cache(maxsize=2)
def OpenClCompileAndLinkFlags(
    opencl_headers: bool = True
) -> typing.Tuple[typing.List[str], typing.List[str]]:
  """Get device-specific OpenCL compile and link flags."""
  cflags, ldflags = _OpenClCompileAndLinkFlags()
  if not opencl_headers:
    cflags = []
  return cflags, ldflags


@functools.lru_cache(maxsize=2)
def LibCeclCompileAndLinkFlags(
    opencl_headers: bool = True
) -> typing.Tuple[typing.List[str], typing.List[str]]:
  """Get device-specific LibCecl compile and link flags."""
  cflags, ldflags = OpenClCompileAndLinkFlags(opencl_headers=opencl_headers)
  return (cflags + ['-isystem', str(LIBCECL_HEADER.parent)],
          ldflags + ['-lcecl', f'-L{LIBCECL_SO.parent}'])
