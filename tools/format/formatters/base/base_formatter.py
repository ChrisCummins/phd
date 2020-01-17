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
import os
import pathlib
import subprocess
import tempfile
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import fasteners
from labm8.py import app
from labm8.py import fs
from tools.format import app_paths

# A deferred formatter action. Calling a formatter may return one of these
# actions. An action is a function which takes no arguments and returns the
# outcome: a list of all paths processed, their original mtimes, and a list of
# exceptions raised during formatting.
FormatAction = Callable[
  [], Tuple[List[pathlib.Path], List[Optional[int]], List[Exception]]
]


class BaseFormatter(object):
  """Base class for implementing formatters.

  This class has two primary usages:
    * Formatting (lots of) files by calling an instance of the class to return
      callbacks to parallelizable actions.
    * Formatting individual strings by calling the static FileFormatter.Format()
      method. This instantiates a formatter, writes the input to a temporary
      file, and formats it, returning the output.

  Subclasses must implement the call operator and Finalize() method. A well
  behaved formatter should have the following properties:

    1. It should reject any malformed input by raising a FormatError exception.
    2. It should reject any well-formed input that requires a modification that
       cannot be made automatically, such as enforing that the capitalization
       of a global variable be changed.
    3. It should write a modified file only if there are changes to be made.
       The format executor uses file mtimes to determine if a file is modified,
       so even if the same contents are written as were read, this will be
       considered a modification.
    4. It should not print to stdout or stderr, although logging using app.Log()
       at log level of 3 and above is permitted.
    5. It should apply a deterministic set of formatting rules. The formatted
       output produced should depend only on the contents of the input, not the
       runtime environment, operating system, etc.
    6. It should test for all required dependencies at construction time, and
       raise an InitError() if anything is missing.
  """

  def __init__(self, cache_path: pathlib.Path):
    """Constructor.

    Subclasses must call this first.

    Args:
      cache_path: The path to the persistent formatter cache.

    Raises:
      InitError: In case a subclass fails to initialize.
    """
    if not cache_path.is_dir():
      raise TypeError(f"Formatter cache not found: {cache_path}")
    self.cache_path = cache_path

    # Lock exclusive inter-process access to all formatters of this type. This
    # lock does not need to be released - cleanup of inter-process locks using
    # the fasteners library is automatic. This will block indefinitely if the
    # lock is already acquired by a different process, ensuring that only a
    # single formatter of this type (name) is running at a time.
    lock_file = self.cache_path / f"{type(self).__name__}.LOCK"
    app.Log(3, "Acquiring lock file: %s", lock_file)
    assert fasteners.InterProcessLock(lock_file)
    app.Log(3, "Lock file acquired: %s", lock_file)

  class FormatError(ValueError):
    """An exception raised during processing of one or more files.

    A format error is not tied to any one path, since the formatter be
    processing multiple paths. The exception message should identify the
    path where appropariate, e.g.

        raise self.FormatError(f"Error formatting: {path}")

    Multiline error messages should have every line except the first indented
    with four spaces to aid in readability when printing to output. E.g.

        raise self.FormatError(f"Error formatting {len(self.paths)} paths:\n"
                               "     " + "\n    ".join(proc.stderr.split("\n"))
    """

  class InitError(TypeError):
    """An error value raised if the formatter fails to initialize.

    This error should only be thrown by the constructor, and should indicate
    some error condition that the formatter cannot recover from..
    """

  def __call__(
    self, path: pathlib.Path, cached_mtime: Optional[int] = None
  ) -> Optional[FormatAction]:
    """Register the path for formatting and return a deferred formatter action.

    Returns:
      A callback which formats the file and raises FormatError on failure.
    """
    raise NotImplementedError

  def Finalize(self) -> Optional[FormatAction]:
    """Finalize the formatter.

    This is a no-op for this class, as there is nothing to finalize since there
    is no batching.
    """
    raise NotImplementedError

  @classmethod
  def Format(cls, text: str, assumed_filename: Optional[str] = None) -> str:
    """Programatically run the formatter on a string of text.

    Args:
      text: The text to format.
      assumed_filename: Use this file name to feed the input to the formatter.

    Returns:
      The formatted text.

    Raises:
      InitError: If the formatter fails to initialize.
      FormatError: If the formatter fails.
    """
    with tempfile.TemporaryDirectory(
      prefix="format_", dir=app_paths.GetCacheDir()
    ) as d:
      cache_path = pathlib.Path(d)
      formatter = cls(cache_path)

      path = cache_path / (assumed_filename or cls.assumed_filename)
      fs.Write(path, text.encode("utf-8"))

      actions = [formatter(path), formatter.Finalize()]

      # Run the actions and propagate errors.
      for action in actions:
        if action:
          _, _, errors = action()
          if errors:
            raise errors[0]

      return fs.Read(path)

  def _Exec(self, cmd: List[str], env: Optional[Dict[str, str]] = None):
    """Run the given command silently.

    This is a utility function for implementing formatters.

    At log level 3 and above, the command that is executed is logged. At log
    level 5 and above, both the command and its output are logged.

    Args:
      cmd: A list of arguments to subprocess.Popen().

    Raises:
      FormatError: If the command fails.
    """
    if app.GetVerbosity() >= 3 and app.GetVerbosity() < 5:
      app.Log(3, "EXEC $ %s", " ".join(str(x) for x in cmd))

    process = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      universal_newlines=True,
      env=env,
    )
    stdout, _ = process.communicate()

    if app.GetVerbosity() >= 5:
      app.Log(5, "EXEC $ %s\n%s", " ".join(str(x) for x in cmd), stdout)

    if process.returncode:
      raise self.FormatError(stdout)

  def _Which(self, name: str, install_instructions: Optional[str] = None):
    """Lookup the absolute path of a binary. If not found, abort.

    This is a utility function for implementing formatters.

    Args;
      name: The binary to look up.

    Returns:
      The abspath of the binary.

    Raises:
      InitError: If the requested name is not found.
    """
    for path in os.environ["PATH"].split(os.pathsep):
      if os.path.exists(os.path.join(path, name)):
        return os.path.join(path, name)

    error = f"Could not find binary required by {type(self).__name__}: {name}"
    if install_instructions:
      error = f"{error}\n{install_instructions}"
    else:
      error = (
        f"{error}\nYou probably haven't installed the development "
        "dependencies. See INSTALL.md."
      )
    raise self.InitError(error)
