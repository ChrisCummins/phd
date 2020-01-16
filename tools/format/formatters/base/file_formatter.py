# Copyright 2020 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This module defines the base classes and utilities for formatters."""
import pathlib
from typing import List
from typing import Optional

from labm8.py import app
from tools.format.formatters.base import base_formatter


class FileFormatter(base_formatter.BaseFormatter):
  """Base class for implementing a single-file formatter.

  Subclasses must implement the RunOne() method, which takes a single path as
  input and formats it. If you are writing a formatter which can handle multiple
  input files in a single run, use the BatchedFileFormatter base class instead.
  """

  # The assumed filename to use when running this formatter with text inputs.
  # Set a custom value for this in formatters which are sensitive to filename,
  # e.g. assumed_file = "text.sql" if a formatter requires inputs to have .sql
  # extensions.
  assumed_filename = "text"

  def __init__(self, *args, **kwargs):
    """Constructor.

    Subclasses must call this first.

    Args:
      args: Positional arguments passed to BaseFormatter.
      kwargs: Keyword arguments passed to BaseFormatter.

    Raises:
      InitError: In case a subclass fails to initialize.
    """
    super(FileFormatter, self).__init__(*args, **kwargs)

  def RunOne(self, path: pathlib.Path) -> None:
    """Format a single file. Subclasses must implement this.

    Args:
      path: The path to a file.

    Raises:
      FormatError: If the formatter fails.
    """
    raise NotImplementedError("abstract class")

  def RunOneWithLog(self, path: pathlib.Path):
    """A wrapper around RunOne() which verbosely logs what's going on."""
    app.Log(2, "%s %s", type(self).__name__, path)
    return self.RunOne(path)

  def __call__(
    self, path: pathlib.Path, cached_mtime: Optional[int] = None
  ) -> Optional[base_formatter.FormatAction]:
    """Register the path for formatting and return a deferred formatter action.

    Returns:
      A callback which formats the file and raises FormatError on failure.
    """
    return lambda: self.Action([path], [cached_mtime])

  def Action(
    self, paths: List[pathlib.Path], cached_mtimes: List[Optional[int]] = None
  ) -> base_formatter.FormatAction:
    """A deferred formatter action.

    Calling this method performs the actual formatting and returns the outcome.

    Returns:
      A tuple of <paths, previous_mtimes, errors> which describes the outcome.
    """
    try:
      self.RunOneWithLog(paths[0])
      return paths, cached_mtimes, []
    except self.FormatError as e:
      return [], [], [e]

  def Finalize(self) -> Optional[base_formatter.FormatAction]:
    """Finalize the formatter.

    This is a no-op for this class, as there is nothing to finalize since there
    is no batching.
    """
    pass
