# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A dataset of the linux source tree.

This is intended to be used for source code analysis, learning from 'big code',
etc.
"""
import pathlib
import subprocess
import typing

from labm8 import app
from labm8 import bazelutil
from labm8 import decorators

FLAGS = app.FLAGS


class LinuxSourcesDataset(object):
  """A dataset of the Linux source tree.

  This is a list of file paths comprising a subset of the linux source tree.
  This is assembled from a fresh clone of the source tree, without configuration
  or building.
  """
  # This value must be changed if the WORKSPACE @linux_srcs archive is changed!
  version = '4.19'

  def __init__(self):
    self._src_tree_root = pathlib.Path(bazelutil.DataPath("linux_srcs"))
    self.generated_hdrs_root = pathlib.Path(
        bazelutil.DataPath("phd/datasets/linux/generated_headers"))

  @property
  def src_tree_root(self) -> pathlib.Path:
    """Get the root of the source tree."""
    return self._src_tree_root

  def ListFiles(self,
                subpath: typing.Union[pathlib.Path, str],
                filename_pattern: str = None):
    """Return a list of paths to files within the source tree.

    Args:
      subpath: The directory to list, relative to the root of the source tree.
      filename_pattern: An optional file pattern to match using `find`, e.g.
        '*.c'.

    Returns:
      A list of paths to files.
    """
    # We can't use '-type f' predicate because bazel symlinks data files.
    # Instead, we'll filter the output of find to remove directories.
    cmd = ['find', str(self.src_tree_root / subpath)]
    if filename_pattern:
      cmd += ['-name', filename_pattern]

    find_output = subprocess.check_output(cmd, universal_newlines=True)

    # The last line of output is an empty line.
    paths = [pathlib.Path(line) for line in find_output.split('\n')[:-1]]

    # Exclude directories from generated output.
    return [p for p in paths if p.is_file()]

  @decorators.memoized_property
  def all_srcs(self) -> typing.List[pathlib.Path]:
    """Return all C files."""
    return self.ListFiles('', '*.c')

  @decorators.memoized_property
  def all_hdrs(self) -> typing.List[pathlib.Path]:
    """Return all header files."""
    return self.ListFiles('', '*.h')

  @decorators.memoized_property
  def kernel_srcs(self) -> typing.List[pathlib.Path]:
    """Return the src files in 'kernel/'."""
    return self.ListFiles('kernel', '*.c')

  @decorators.memoized_property
  def kernel_hdrs(self) -> typing.List[pathlib.Path]:
    """Return the header files in 'kernel/'."""
    return self.ListFiles('kernel', '*.h')

  @property
  def kernel_srcs_and_hdrs(self):
    """Return a concatenation of the 'kernel/' sources and headers."""
    return self.kernel_srcs + self.kernel_hdrs
