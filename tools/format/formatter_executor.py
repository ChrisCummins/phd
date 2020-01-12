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
import glob
import multiprocessing
import os
import pathlib
import queue
import threading
from typing import List

import sqlalchemy as sql

from labm8.py import app
from labm8.py import crypto
from labm8.py import shell
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
    self.errors = False

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
    with concurrent.futures.ThreadPoolExecutor(
      max_workers=multiprocessing.cpu_count()
    ) as executor, self.DatabaseConnection() as connection:
      futures = []

      while True:
        path = self.q.get(block=True)

        # A none value means that they we have run out of paths to enumerate.
        if path is None:
          break

        # Determine if the file should be processed.
        mtime = int(os.path.getmtime(path) * 1e6)
        cached_mtime = None
        if FLAGS.with_cache:
          query = connection.execute(
            sql.text("SELECT mtime FROM cache WHERE path = :path"),
            path=str(path.absolute()),
          )
          result = query.first()
          if result:
            cached_mtime = result[0]
        # Skip
        if mtime == cached_mtime:
          continue

        # Get or create the formatter.
        key = path.suffix or path.name
        if key in self.formatters:
          form = self.formatters[key]
        else:
          form = formatters[key](self.cache_path)
          self.formatters[key] = form

        # Run the formatter.
        action = form(path, cached_mtime)
        if action:
          futures.append(executor.submit(action))

      for form in self.formatters.values():
        action = form.Finalize()
        if action:
          futures.append(executor.submit(action))

      for future in futures:
        paths, cached_mtimes, error = future.result()
        for path, cached_mtime in zip(paths, cached_mtimes):
          mtime = int(os.path.getmtime(path) * 1e6)
          if mtime != cached_mtime:
            print(path)
            if not error and FLAGS.with_cache:
              connection.execute(
                sql.text(
                  "REPLACE INTO cache (path, mtime) VALUES (:path, :mtime)"
                ),
                path=str(path.absolute()),
                mtime=mtime,
              )

        if error:
          print(error, file=sys.stderr)
          self.errors = True
        elif FLAGS.with_cache:
          # TODO: Run as a single query.
          mtime = int(os.path.getmtime(path) * 1e6)
