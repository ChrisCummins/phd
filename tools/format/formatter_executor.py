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
"""This module defines the master thread for executing formatters."""
import concurrent.futures
import contextlib
import multiprocessing
import os
import pathlib
import queue
import sys
import threading

import sqlalchemy as sql

from labm8.py import app
from labm8.py import sqlutil
from tools.format.formatters.suffix_mapping import mapping as formatters


FLAGS = app.FLAGS


class FormatterExecutor(threading.Thread):
  """This thread reads from a queue of paths and executes formatters on them.

  This thread uses a database of "last modified" timestamps.

  It reads from a queue of paths to process, dispatching the paths to specifid
  formatters, and writing the results to the cache.
  """

  def __init__(self, cache_path: pathlib.Path, q: queue.Queue):
    """Constructor.

    Args:
      cache_path: The cache directory.
      q: A queue of paths to process. All paths are assumed to: (a) exist, (b)
        be unique.
    """
    super(FormatterExecutor, self).__init__()
    self.cache_path = cache_path
    self.q = q
    self.formatters = {}

    # The results of formatting. Access these member variables after the thread
    # has terminated.
    self.errors = False
    self.modified_files = []

  @contextlib.contextmanager
  def DatabaseConnection(self):
    """A scoped database connection."""
    engine = sqlutil.CreateEngine(
      f"sqlite:///{self.cache_path}/cache.sqlite.db"
    )

    # Create the table.
    engine.execute(
      """
CREATE TABLE IF NOT EXISTS cache(
  path VARCHAR(4096) NOT NULL PRIMARY KEY,
  mtime INTEGER NOT NULL
);
"""
    )

    connection = engine.connect()
    transaction = connection.begin()
    try:
      yield connection
      transaction.commit()
    except:
      transaction.rollback()
      raise
    finally:
      transaction.close()
      connection.close()

  def run(self):
    """Read the input paths and format them as required.."""
    with concurrent.futures.ThreadPoolExecutor(
      max_workers=multiprocessing.cpu_count()
    ) as executor, self.DatabaseConnection() as connection:
      futures = []

      while True:
        path = self.q.get(block=True)

        # A none value means that they we have run out of paths to enumerate.
        if path is None:
          break

        action = self.MaybeFormat(connection, path)
        if action:
          futures.append(executor.submit(action))

      # We have run out of paths to format, finalize the formatters.
      for form in self.formatters.values():
        action = form.Finalize()
        if action:
          futures.append(executor.submit(action))

      # Wait for the formatters to complete.
      for future in concurrent.futures.as_completed(futures):
        paths, cached_mtimes, error = future.result()

        for path, cached_mtime in zip(paths, cached_mtimes):
          mtime = int(os.path.getmtime(path) * 1e6)
          if mtime != cached_mtime:
            print(path)
            self.modified_files.append(path)
            if not error and FLAGS.with_cache:
              connection.execute(
                sql.text(
                  "REPLACE INTO cache (path, mtime) VALUES (:path, :mtime)"
                ),
                path=str(path),
                mtime=mtime,
              )

        if error:
          print(error, file=sys.stderr)
          self.errors = True

  def MaybeFormat(self, connection, path: pathlib.Path):
    """Schedule a file to be formatted if required."""
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
    if mtime == cached_mtime:
      return

    # Get or create the formatter.
    key = path.suffix or path.name
    if key in self.formatters:
      form = self.formatters[key]
    else:
      form = formatters[key](self.cache_path)
      self.formatters[key] = form

    return form(path, cached_mtime)
