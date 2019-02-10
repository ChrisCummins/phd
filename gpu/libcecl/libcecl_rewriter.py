"""Convert code to-and-from using OpenCL and libcecl."""
import collections
import pathlib
import typing

from absl import app
from absl import flags
from absl import logging

from labm8 import fs


FLAGS = flags.FLAGS

flags.DEFINE_list('opencl_rewrite_paths', None,
                  'A list of paths to rewrite using libcecl.')
flags.DEFINE_list('libcecl_rewrite_paths', None,
                  'A list of paths to rewrite using OpenCL.')

Rewrite = collections.namedtuple('Rewrite', ('opencl', 'libcecl'))

# The OpenCL functions and their corresponding libcecl implementations. We use
# a list of rewrites rather than a map since the order that they are applied
# is significant.
OPENCL_TO_LIBCECL_REWRITES = [
  Rewrite('clBuildProgram', 'CECL_PROGRAM'),
  Rewrite('clCreateBuffer', 'CECL_BUFFER'),
  Rewrite('clCreateCommandQueue', 'CECL_CREATE_COMMAND_QUEUE'),
  Rewrite('clCreateKernel', 'CECL_KERNEL'),
  Rewrite('clCreateProgramWithSource', 'CECL_PROGRAM_WITH_SOURCE'),
  Rewrite('clEnqueueMapBuffer', 'CECL_MAP_BUFFER'),
  Rewrite('clEnqueueNDRangeKernel', 'CECL_ND_RANGE_KERNEL'),
  Rewrite('clEnqueueReadBuffer', 'CECL_READ_BUFFER'),
  Rewrite('clEnqueueTask', 'CECL_TASK'),
  Rewrite('clEnqueueWriteBuffer', 'CECL_WRITE_BUFFER'),
  Rewrite('clSetKernelArg', 'CECL_SET_KERNEL_ARG'),
  Rewrite('clCreateContextFromType', 'CECL_CREATE_CONTEXT_FROM_TYPE'),
  Rewrite('clCreateContext', 'CECL_CREATE_CONTEXT'),
  Rewrite('clGetKernelWorkGroupInfo', 'CECL_GET_KERNEL_WORK_GROUP_INFO'),
]


def RewriteOpenClSource(src: str) -> str:
  """Replace OpenCL calls with libcecl."""
  for rewrite in OPENCL_TO_LIBCECL_REWRITES:
    src = src.replace(rewrite.opencl, rewrite.libcecl)
  # Prepend the libcecl header.
  src = f'#include <libcecl.h>\n{src}'
  return src


def RewriteLibceclSource(src: str) -> str:
  """Replace libcecl calls with OpenCL."""
  for rewrite in reversed(OPENCL_TO_LIBCECL_REWRITES):
    src = src.replace(rewrite.libcecl, rewrite.opencl)
  # Strip the libcecl include.
  src = src.replace('#include <libcecl.h>\n', '')
  return src


def RewriteOpenClFileInPlace(path: pathlib.Path) -> None:
  """Re-write a file using OpenCL in-place."""
  with open(path) as f:
    src = f.read()
  with open(path, 'w') as f:
    f.write(RewriteOpenClSource(src))


def RewriteLibceclFileInPlace(path: pathlib.Path) -> None:
  """Re-write a file using libcecl in-place."""
  with open(path) as f:
    src = f.read()
  with open(path, 'f') as f:
    f.write(RewriteLibceclSource(src))


def FileShouldBeRewritten(path: pathlib.Path) -> bool:
  """Return whether a file should be re-written."""
  return path.suffix in {'.c', '.cc', '.cxx', '.cpp', '.h', '.hpp'}


def GetFilesToRewriteFromPath(
    path: pathlib.Path) -> typing.Iterable[
  pathlib.Path]:
  """Get an iterator of files to rewrite."""

  def _EnumerateFiles():
    if path.is_file():
      yield path
    else:
      for path_ in fs.ls(path, recursive=True, abspaths=True):
        yield pathlib.Path(path_)

  return (path for path in _EnumerateFiles() if FileShouldBeRewritten(path))


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  for path in FLAGS.opencl_rewrite_paths:
    for path in GetFilesToRewriteFromPath(pathlib.Path(path)):
      logging.info('%s', path)
      RewriteOpenClFileInPlace(path)
  for path in FLAGS.libcecl_rewrite_paths:
    for path in GetFilesToRewriteFromPath(pathlib.Path(path)):
      logging.info('%s', path)
      RewriteLibceclFileInPlace(path)


if __name__ == '__main__':
  app.run(main)
