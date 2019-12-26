# Copyright 2019 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
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
"""This module defines a RUN_ID variable that uniquely identifies a program
invocation.

When executed as a script, this prints a run ID.

Usage:

    $ bazel run //deeplearning/ml4pl:run_id
    run_id:191129T133441:example
"""
import datetime
import os
import pathlib
import re
import sys
import time
from typing import NamedTuple
from typing import Union

import fasteners
import sqlalchemy as sql

from deeplearning.ml4pl import filesystem_paths
from labm8.py import app
from labm8.py import fs
from labm8.py import system


FLAGS = app.FLAGS

_RUN_ID_LOCK_PATH = filesystem_paths.TemporaryFilePath("run_id.lock")


RUN_ID_MAX_LEN: int = 40
RUN_ID_REGEX = re.compile(
  r"(?P<script_name>[A-Za-z0-9_]+):"
  r"(?P<timestamp>[0-9]{12}):"
  r"(?P<hostname>[A-Za-z0-9]+)"
)


class RunId(NamedTuple):
  """A run ID.

  A run ID is a <= 40 character string with the format:

      <script_name>:<timestamp>:<hostname>

  Where <script_name> is a <= 16 char script stem, <timestamp> is a UTC-formatted
  timestamp with seconds precision, and <hostname> is a <= 12 char system
  hostname.
  """

  script_name: str  # max 16 chars
  timestamp: str  # 12 chars
  hostname: str  # max 12 chars

  @property
  def datetime(self) -> datetime.datetime:
    """Return the timestamp as a datetime."""
    return datetime.datetime.strptime(self.timestamp, "%y%m%d%H%M%S")

  def __repr__(self):
    """Stringify the run ID."""
    return f"{self.script_name}:{self.timestamp}:{self.hostname}"

  def __eq__(self, rhs: Union["RunId", str]):
    """Equivalency check."""
    return str(self) == str(rhs)

  @classmethod
  def FromString(cls, run_id: str) -> "RunId":
    """Construct a run ID from string.

    Args:
      run_id: The string run ID.

    Returns:
      A RunId instance.

    Raises:
      ValueError: If the run ID is malformed or invalid.
    """
    match = RUN_ID_REGEX.match(run_id)
    if not match:
      raise ValueError(f"Invalid run ID: '{run_id}'")
    if len(match.group("script_name")) > 16:
      raise ValueError(
        f"Script name exceeds 16 char: {match.group('script_name')}"
      )
    if len(match.group("timestamp")) != 12:
      raise ValueError(f"Timestamp is not 12 char: {match.group('timestamp')}")
    if len(match.group("hostname")) > 12:
      raise ValueError(f"Hostname exceeds 12 char: {match.group('hostname')}")
    return cls(
      match.group("script_name"),
      match.group("timestamp"),
      match.group("hostname"),
    )

  @classmethod
  def SqlStringColumn(
    cls, default=lambda: str(RUN_ID), index: bool = True
  ) -> sql.Column:
    """Generate an SQLAlchemy column which stores run IDs as strings.

    Args:
      default: Set a default value for the run ID. Use a lambda to defer
        evaluation of the run ID, preventing it from being hardcoded in the SQL
        table CREATE statement.
    """
    return sql.Column(
      cls.SqlStringColumnType(), nullable=False, default=default, index=index
    )

  @staticmethod
  def SqlStringColumnType() -> sql.String:
    """Generate a SQL column type to store run IDs as strings."""
    return sql.String(RUN_ID_MAX_LEN)

  @classmethod
  def GenerateGlobalUnique(cls) -> "RunId":
    """Generate a unique run ID for this script. Don't call this method, use
    RUN_ID instead.

    The uniqueness of run IDs is provided by storing the most-recently generated
    run ID in /tmp/ml4pl_previous_run_id.txt. If multiple jobs all start at the
    same time, their timestamps will collide, so this script will wait until the
    run ID is unique. Note the implementation is not truly atomic - there is
    still an incredibly slight possibility of generating duplicate run IDs if
    enough concurrent jobs attempt to grab run IDs at the same time.

    If required, the run ID can be forced to a specific value by setting a
    RUN_ID environment variable. This may be useful for debugging, but is
    generally a bad idea.

    Returns:
      A run ID instance for this script invocation.
    """
    # Optionally allow a forced run ID using an environment variable. We can't
    # use a flag here as this method at module import time, before flags are
    # parsed.
    forced_run_id = os.environ.get("RUN_ID")
    if forced_run_id:
      return forced_run_id

    script_name = pathlib.Path(sys.argv[0]).stem

    return cls.GenerateUnique(script_name)

  @classmethod
  def GenerateUnique(cls, name: str) -> "RunId":
    """Generate a unique run ID with the given name.

    The uniqueness of run IDs is provided by storing the most-recently generated
    run ID in /tmp/ml4pl_previous_run_id.txt. If run IDs are requested at the
    same time, their timestamps will collide, and this method will block until
    the run ID is unique. Note the implementation is not truly atomic - there is
    still an incredibly slight possibility of generating duplicate run IDs if
    enough concurrent jobs attempt to grab run IDs at the same time.

    Args:
      name: The name of the run ID to generate.

    Returns:
      A run ID instance.
    """
    # Truncate the name if required.
    name = name[:16]

    # Compute a new run ID.
    timestamp = time.strftime("%y%m%d%H%M%S")
    hostname = system.HOSTNAME[:12]
    run_id = f"{name}:{timestamp}:{hostname}"

    # Begin inter-process locked region.
    with fasteners.InterProcessLock(_RUN_ID_LOCK_PATH):
      # Check if there is already a run with this ID.
      previous_run_id_path = filesystem_paths.TemporaryFilePath(
        f"previous_run_id_{name}.txt"
      )
      previous_run_id = None
      if previous_run_id_path.is_file():
        previous_run_id = fs.Read(previous_run_id_path)

      # The run ID is unique, so write it and return it.
      if run_id != previous_run_id:
        previous_run_id_path.parent.mkdir(exist_ok=True, parents=True)
        fs.Write(previous_run_id_path, run_id.encode("utf-8"))
        return RunId.FromString(run_id)
    # End of inter-process locked region.

    # We didn't get a unique run ID, so try again.
    time.sleep(0.2)
    return cls.GenerateGlobalUnique()


# The public variables:
RUN_ID: RunId = RunId.GenerateGlobalUnique()


def Main():
  """Main entry point."""
  print(RUN_ID)


if __name__ == "__main__":
  app.Run(Main)
