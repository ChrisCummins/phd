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
"""This module defines the Format class for formatting files."""
import concurrent.futures
import multiprocessing
import os
import pathlib
import queue
import sys
import threading
from typing import Dict
from typing import Iterable
from typing import List
from typing import Union

import sqlalchemy as sql
from labm8.py import app
from labm8.py import ppar
from labm8.py import shell
from labm8.py import sqlutil
from tools.format import app_paths
from tools.format.default_suffix_mapping import (
  mapping as default_suffix_mapping,
)
from tools.format.formatters.base import base_formatter


FLAGS = app.FLAGS

app.DEFINE_boolean(
  "with_cache",
  True,
  'Enable the persistent caching of "last modified" timestamps for files. '
  "Files which have not changed since the last time the formatter was run are "
  "skipped. Running the formatter with --nowith_cache forces all files to be "
  "formatted, even if they have not changed.",
)


class FormatPaths(object):
  """An object which iterates over an input sequence of paths and formats them.

  For each path input:
    1. Lookup whether the path has an associated formatter for it.
    2. Lookup the mtime of the path, and determine whether it has been modified
       since the last time that the formatter processed this path.
    3. Create (or-reuse) a formatter for this type.
    4. Tell the formatter to format this path.
    5. Record the action (if any) that the formatter must perform.

  This object then manages the execution of formatter actions, overlapping their
  asynchronous execution through thread-level parallelism, and caches the mtime
  of visited files.

  An instance of this formatter can be iterated over to receive a stream of
  formatting outcomes. A formatting outcome is either a pathlib.Path of a
  modified file, or the Exception instance of an error that was encountered.
  Files that are visited but not modified are not returned by this iterator.
  Example usage:

      for outcome in FormatPaths(paths_to_format):
        if isinstance(outcome, Exception):
          print("ERROR:", outcome)
        else:
          print("Modified file:", outcome)

  If you don't want to iterate over the results, you can instead use the Join()
  method to block until all formatting actions have completed:

    formatter = FormatPaths(paths_to_format)
    formatter.Join()
  """

  def __init__(
    self,
    paths: Iterable[pathlib.Path],
    dry_run: bool = False,
    suffix_mapping: Dict[
      str, base_formatter.BaseFormatter
    ] = default_suffix_mapping,
  ):
    """Constructor.

    Args:
      dry_run: Yield only the paths that would be formatted, without running the
        formatters themselves.
    """
    self.dry_run = dry_run
    self.suffix_mapping = suffix_mapping

    # A lazily-instantiated mapping from formatter class name to formatter
    # instance. Each key is the __name__ of a class from the
    # suffix_mapping dictionary. We use the class name rather than
    # the file suffix to index into this map since there may be a many-to-one
    # relationship of suffix -> formatter, and we want to insure that only a
    # single formatter of each type is instantiated.
    self.formatter_instances = {}

    self.cache_path = app_paths.GetCacheDir()

    # A queue of elements to return from the iterator.
    self._queue = queue.Queue()

    # Create a threaded iterator to filter the list of incoming paths.
    paths = ppar.ThreadedIterator(
      PathsWithFormatters(paths, self.suffix_mapping), max_queue_size=0
    )

    self._thread = threading.Thread(target=lambda: self.Run(paths))
    self._thread.start()

  def GetDatabaseEngine(self) -> sql.engine.Engine:
    """Construct the database engine."""
    engine = sqlutil.CreateEngine(
      f"sqlite:///{self.cache_path}/cache.sqlite.db"
    )

    engine.execute(
      """
    CREATE TABLE IF NOT EXISTS cache(
      path VARCHAR(4096) NOT NULL PRIMARY KEY,
      mtime INTEGER NOT NULL
    );
    """
    )

    return engine

  def Run(
    self, paths: Iterable[pathlib.Path]
  ) -> Iterable[Union[pathlib.Path, Exception]]:
    """Read the input paths and format them as required.."""
    with concurrent.futures.ThreadPoolExecutor(
      max_workers=multiprocessing.cpu_count()
    ) as executor, self.GetDatabaseEngine().begin() as connection:
      # A list of
      futures: List[concurrent.futures.Future] = []

      try:
        for path in paths:
          needs_formatting, cached_mtime = self.NeedsFormatting(
            connection, path
          )
          if needs_formatting:
            if self.dry_run:
              self._queue.put(path)
            else:
              try:
                action = self.MaybeFormat(path, cached_mtime)
                if action:
                  futures.append(executor.submit(action))
              except base_formatter.BaseFormatter.InitError as init_error:
                # If a formatter failed to initialize then record the error and
                # stop what we're doing.
                self._queue.put(init_error)
                break

        # We have run out of paths to format so finalize the formatters we
        # instantiated.
        for form in self.formatter_instances.values():
          action = form.Finalize()
          if action:
            futures.append(executor.submit(action))

        # Wait for the formatters to complete.
        for future in concurrent.futures.as_completed(futures):
          paths, cached_mtimes, errors = future.result()

          # Accumulate the errors which can be checked later.
          for error in errors:
            self._queue.put(error)

          for path, cached_mtime in zip(paths, cached_mtimes):
            mtime = int(os.path.getmtime(path) * 1e6)
            if mtime != cached_mtime:
              self._queue.put(path)
              if FLAGS.with_cache:
                connection.execute(
                  sql.text(
                    "REPLACE INTO cache (path, mtime) VALUES (:path, :mtime)"
                  ),
                  path=str(path),
                  mtime=mtime,
                )
      finally:
        # End of input loop, terminate.
        self._queue.put(None)

  def __iter__(self) -> Iterable[Union[pathlib.Path, Exception]]:
    """Return an iterator over modification outcomes."""
    while True:
      item = self._queue.get()
      if item is None:
        break
      yield item

    self.Join()

  def Join(self):
    self._thread.join()

  def NeedsFormatting(self, connection, path: pathlib.Path):
    # Determine if the file should be processed.
    mtime = int(os.path.getmtime(path) * 1e6)
    cached_mtime = None
    if FLAGS.with_cache:
      query = connection.execute(
        sql.text("SELECT mtime FROM cache WHERE path = :path"), path=str(path),
      )
      result = query.first()
      if result:
        cached_mtime = result[0]
    # Skip a file that hasn't been modified since the last time it was
    # formatted.
    return mtime != cached_mtime, cached_mtime

  def MaybeFormat(self, path: pathlib.Path, cached_mtime):
    """Schedule a file to be formatted if required.

    Raises:
      formatter.Formatter.InitError: If a formatter fails to initialize.
    """
    # Get or create the formatter.
    formatters_key = path.suffix or path.name
    # Lookup the name of the formatter class, which is used to index into the
    # formatter instance cache.
    formatter_cache_key = self.suffix_mapping[formatters_key].__name__
    if formatter_cache_key in self.formatter_instances:
      form = self.formatter_instances[formatter_cache_key]
    else:
      form = self.suffix_mapping[formatters_key](self.cache_path)
      self.formatter_instances[formatter_cache_key] = form

    return form(path, cached_mtime)


def PathsWithFormatters(
  paths: Iterable[pathlib.Path],
  suffix_mapping: Dict[str, base_formatter.BaseFormatter],
) -> Iterable[pathlib.Path]:
  """Filter an iterable of paths for those with corresponding formatters.

  Args:
    paths: An iterator of paths.

  Returns:
    An iterator of paths which have entries in the suffix_mapping.formatters
    dictionary.
  """
  for path in paths:
    key = path.suffix or path.name
    if key in suffix_mapping:
      app.Log(3, "PUT %s", path)
      yield path


def FormatPathsOrDie(
  paths: Iterable[pathlib.Path], dry_run: bool = False
) -> List[pathlib.Path]:
  """Run the formatting loop and terminate on error.

  This runs the formatting loop, print the paths of any modified paths as they
  are processed. Any formatting errors cause this function to crash the process
  with an error.
  """
  # Accumulate errors which we print at the end.
  errors = []
  # Accumulate modified paths that are returned at the end.
  modified_paths = []

  for outcome in FormatPaths(paths, dry_run=dry_run):
    if isinstance(outcome, Exception):
      errors.append(outcome)
    else:
      modified_paths.append(outcome)
      print(outcome)

  if errors:
    print(
      f"{shell.ShellEscapeCodes.RED}"
      "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      f"{shell.ShellEscapeCodes.END}",
      file=sys.stderr,
    )
    print(
      f"{shell.ShellEscapeCodes.RED}"
      f"Formatting completed with {len(errors)} errors"
      f"{shell.ShellEscapeCodes.END}",
      file=sys.stderr,
    )
    for i, error in enumerate(errors, start=1):
      print(
        f"{shell.ShellEscapeCodes.RED}ERROR {i}:{shell.ShellEscapeCodes.END}",
        error,
        file=sys.stderr,
      )
    print(
      f"{shell.ShellEscapeCodes.RED}"
      "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
      f"{shell.ShellEscapeCodes.END}",
      file=sys.stderr,
    )
    sys.exit(12)

  return modified_paths
