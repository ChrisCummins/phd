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
from tools.format.formatters.base import file_formatter


FLAGS = app.FLAGS


class BatchedFileFormatter(file_formatter.FileFormatter):
  """A formatter which processes multiple files in a single run.

  Instead of requiring a RunOne() method to be implemented, you must instead
  provide a RunMany() method which takes a list of paths and formats them.
  Use this subclass for formatters which can handle multiple files
  simultaneously.
  """

  # The maximum number of files to run in a single pass.
  max_batch_size = 32

  def __init__(self, cache_path: pathlib.Path):
    """Constructor.

    Subclasses must call this first.

    Args:
      cache_path: The path to the persistent formatter cache.

    Raises:
      InitError: In case a subclass fails to initialize.
    """
    super(BatchedFileFormatter, self).__init__(cache_path)

    # A batch of paths and their mtimes to format in a single call to RunMany().
    # These lists will grow up to max_batch_size and then be processed.
    self._paths: List[pathlib.Path] = []
    self._cached_mtimes: List[int] = []

  def RunMany(self, paths: List[pathlib.Path]) -> None:
    """Process a list of files.

    Args:
      paths: A list of files.

    Raises:
      FormatError: In case formatting fails on any of the files.
    """
    raise NotImplementedError

  def __call__(
    self, path: pathlib.Path, cached_mtime: Optional[int] = None
  ) -> base_formatter.FormatAction:
    """Register the path for formatting.

    This buffers the paths to format and executes a single run of RunMany() once
    max_batch_size paths have been buffered.

    Returns:
      A callback which formats a batch of files and raises FormatError on
      failure.
    """
    self._paths.append(path)
    self._cached_mtimes.append(cached_mtime)

    if len(self._paths) > self.max_batch_size:
      return self.Finalize()

  def RunManyWithLog(self, paths: List[pathlib.Path]):
    """A wrapper around RunMany() which verbosely logs what's going on."""
    if app.GetVerbosity() >= 2:
      app.Log(2, "%s %s", type(self).__name__, " ".join(str(x) for x in paths))
    return self.RunMany(paths)

  def Action(
    self, paths: List[pathlib.Path], cached_mtimes: List[int]
  ) -> base_formatter.FormatAction:
    """A deferred formatter action.

    Calling this method performs the actual formatting and returns the outcomes.

    Returns:
      A tuple of <paths, previous_mtimes, errors> which describes the outcomes.
    """
    try:
      self.RunManyWithLog(paths)
    except self.FormatError as e:
      # Because multiple files have been processed, and a failure may be caused
      # by any one (or multiple) of them, we perform a divide-and-conquer on
      # error to isolate only the files that cannot be formatted. This maximises
      # the amount of work that can be usefully performed in a single pass of
      # the formatter.
      #
      # For example, let's say we have a batch of four Python files to process:
      #
      #     a.py
      #     b.py
      #     c.py
      #     d.py  # contains a syntax error which cannot be formatted.
      #
      # When this method is called, RunMany() is called first with all four
      # files, raising an error. We then partition the paths into two groups,
      # [a.py, b.py], and [c.py, d.py], where the first group now succeeds and
      # the second group. Further subdividing the second group causes c.py to
      # now be formatted succesfully, isolating the error at d.py. In this case,
      # [a.py, b.py, c.py] would all be returned as succesfully modified paths,
      # and the error from running d.py in isolation would be returned.
      #
      # Depending on the implementation of RunMany(), this may cause a file to
      # be redundantly formatted up to Log(n) - 1 times, if a "failing" run
      # still formats the valid files.
      if len(paths) == 1:
        return [], [], [e]
      else:
        app.Log(
          2, "EXCEPT on %s with %s files", type(self).__name__, len(paths)
        )
        mid = len(paths) // 2
        left = self.Action(paths[:mid], cached_mtimes[:mid])
        right = self.Action(paths[mid:], cached_mtimes[mid:])
        return tuple(l + r for l, r in zip(left, right))
    return paths, cached_mtimes, []

  def Finalize(self) -> Optional[base_formatter.FormatAction]:
    """Finalize the formatter.

    This flushes the buffer of files that have yet to be formatted.

    Returns:
      A formatter action, if there any files are still pending.
    """
    if self._paths:
      paths = self._paths.copy()
      cached_mtimes = self._cached_mtimes.copy()
      self._paths, self._cached_mtimes = [], []
      return lambda: self.Action(paths, cached_mtimes)
