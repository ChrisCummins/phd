"""A standalone module of utility code for linting.

Code in this file can have no phd dependencies, and is invoked directly, not
through bazel.
"""
from __future__ import print_function

import itertools
import os
import subprocess
import sys
import threading

# The path to the root of the PhD repository, i.e. the directory which this file
# is in.
# WARNING: Moving this file may require updating this path!
_PHD_ROOT = os.path.join(
  os.path.dirname(os.path.realpath(__file__)), "../../.."
)

# The maximum number of arguments to pass to a single binary invocation.
MAX_ARGUMENT_LIST_LENGTH = 128


def WhichOrDie(name):
  """Lookup the absolute path of a binary. If not found, abort.

  Args;
    name: The binary to look up.

  Returns:
    The abspath of the binary.
  """
  for path in os.environ["PATH"].split(os.pathsep):
    if os.path.exists(os.path.join(path, name)):
      return os.path.join(path, name)
  print(
    sys.argv[0], "error: Could not find required binary:", name, file=sys.stderr
  )
  print(
    "You probably haven't installed the development dependencies. "
    "See INSTALL.md.",
    file=sys.stderr,
  )
  sys.exit(1)


BUILDIFIER = WhichOrDie("buildifier")
CLANG_FORMAT = WhichOrDie("clang-format")
GO = WhichOrDie("go")
JAVA = WhichOrDie("java")
JSBEAUTIFY = WhichOrDie("js-beautify")
JSON_LINT = WhichOrDie("jsonlint")
SQLFORMAT = WhichOrDie("sqlformat")
BLACK = WhichOrDie("black")
REORDER_PYTHON_IMPORTS = WhichOrDie("reorder-python-imports")

JSBEAUTIFY_RC = os.path.join(_PHD_ROOT, "tools/code_style/jsbeautifyrc.json")
assert os.path.isfile(JSBEAUTIFY_RC)
GOOGLE_JAVA_FORMAT = os.path.join(
  _PHD_ROOT, "tools/code_style/linters/google-java-format-1.7-all-deps.jar"
)
assert os.path.isfile(GOOGLE_JAVA_FORMAT)


def Print(*args, **kwargs):
  """A print() wrapper that flushes output. Used to prevent line buffering."""
  print(*args, **kwargs)
  sys.stdout.flush()


def ExecOrDie(cmd):
  """Run the given command and return the output.

  Both stdout and stderr are captured. If the command fails, the stderr is
  printed and the process terminates.

  Args:
    cmd: The command to execute, as a list of strings.

  Returns:
     The process stdout as a string.
  """
  process = subprocess.Popen(
    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
  )
  stdout, stderr = process.communicate()
  if process.returncode:
    Print("error")
    Print("    $ ", " ".join(cmd), sep="")

    for line in stderr.rstrip().split("\n"):
      Print("   ", line)

    sys.exit(1)
  return stdout


def GetGitBranchOrDie():
  """Get the name of the current git branch."""
  branches = ExecOrDie(["git", "branch"])
  for line in branches.split("\n"):
    if line.startswith("* "):
      return line[2:]
  print("fatal: Unable to determine git branch", file=sys.stderr)
  sys.exit(1)


def GetGitRemoteOrDie(branch_name):
  """Get the name of the current git remote.

  If there is no remote configured for the given branch, returns None.
  """
  try:
    fully_qualified_name = subprocess.check_output(
      ["git", "rev-parse", "--abbrev-ref", branch_name + "@{upstream}"],
      stderr=subprocess.PIPE,
      universal_newlines=True,
    )
    components = fully_qualified_name.split("/")
    assert len(components) > 1
    return components[0]
  except subprocess.CalledProcessError:
    return None


def Chunkify(iterable, chunk_size):
  """Split an iterable into chunks of a given size.

  This is copied from labm8.py.labtypes.Chunkify().

  Args:
    iterable: The iterable to split into chunks.
    chunk_size: The size of the chunks to return.

  Returns:
    An iterator over chunks of the input iterable.
  """
  i = iter(iterable)
  piece = list(itertools.islice(i, chunk_size))
  while piece:
    yield piece
    piece = list(itertools.islice(i, chunk_size))


class LinterThread(threading.Thread):
  def __init__(self, paths, verbose=False):
    super(LinterThread, self).__init__()
    assert paths
    self._original_mtimes = [os.path.getmtime(f) for f in paths]
    self._paths = paths
    self._verbose = verbose

  @property
  def paths(self):
    return self._paths

  @property
  def modified_paths(self):
    modified_paths = []
    for path, original_mtime in zip(self.paths, self._original_mtimes):
      if os.path.getmtime(path) != original_mtime:
        modified_paths.append(path)
    return modified_paths

  def run(self):
    for chunk in Chunkify(self._paths, MAX_ARGUMENT_LIST_LENGTH):
      self._WrapRunMany(chunk)

  def _WrapRunMany(self, paths):
    """Lint multiple files."""
    if self._verbose:
      print(type(self).__name__, paths)
    self.RunMany(paths)

  def _WrapRunOne(self, path):
    if self._verbose:
      print(type(self).__name__, path)
    self.RunOne(path)

  def RunMany(self, paths):
    """Lint multiple files."""
    for path in paths:
      self._WrapRunOne(path)

  def RunOne(self, path):
    """Lint a single file."""
    self._WrapRunMany([path])


class BuildiferThread(LinterThread):
  def RunMany(self, paths):
    ExecOrDie([BUILDIFIER] + paths)


class ClangFormatThread(LinterThread):
  def RunMany(self, paths):
    # TODO(cec): Use project-local clang-format style file.
    ExecOrDie([CLANG_FORMAT, "-style", "Google", "-i"] + paths)


class PythonThread(LinterThread):
  def RunMany(self, paths):
    ExecOrDie([BLACK, "--line-length=80", "--target-version=py37"] + paths)
    ExecOrDie([REORDER_PYTHON_IMPORTS, "--exit-zero-even-if-changed"] + paths)


class SqlFormat(LinterThread):
  def RunOne(self, path):
    self.Lint(path)

  @staticmethod
  def Lint(path):
    ExecOrDie(
      [
        SQLFORMAT,
        "--reindent",
        "--keywords",
        "upper",
        "--identifiers",
        "lower",
        path,
        "--outfile",
        path,
      ]
    )


class JsBeautifyThread(LinterThread):
  def RunMany(self, paths):
    ExecOrDie([JSBEAUTIFY, "--replace", "--config", JSBEAUTIFY_RC] + paths)


class GoFmtThread(LinterThread):
  def RunOne(self, path):
    # Run linter on each file individually because:
    #   1. An error in one file prevents linting in all other files.
    #   2. All files in a single invocation must be in the same directory.
    ExecOrDie([GO, "fmt", path])


class GoogleJavaFormatThread(LinterThread):
  def RunMany(self, paths):
    ExecOrDie([JAVA, "-jar", GOOGLE_JAVA_FORMAT, "-i"] + paths)


class JsonlintThread(LinterThread):
  def RunOne(self, path):
    ExecOrDie([JSON_LINT, "-i", path])


class LinterActions(object):
  def __init__(self, paths, verbose=False):
    self._paths = paths
    self._buildifier = []
    self._clang_format = []
    self._python = []
    self._sqlformat = []
    self._jsbeautify = []
    self._gofmt = []
    self._java = []
    self._json = []
    self._modified_paths = []
    self._verbose = verbose

    for path in paths:
      basename = os.path.basename(path)
      if basename == "BUILD" or basename == "WORKSPACE":
        self._buildifier.append(path)

      _, extension = os.path.splitext(path)

      if (
        extension == ".cc"
        or extension == ".c"
        or extension == ".h"
        or extension == ".ino"
      ):
        self._clang_format.append(path)
      elif extension == ".py" or extension == ".bzl":
        self._python.append(path)
      elif extension == ".sql":
        self._sqlformat.append(path)
      elif extension == ".html" or extension == ".css" or extension == ".js":
        self._jsbeautify.append(path)
      elif extension == ".go":
        self._gofmt.append(path)
      elif extension == ".java":
        self._java.append(path)
      elif extension == ".json":
        self._json.append(path)

  @property
  def paths(self):
    return self._paths

  @property
  def paths_with_actions(self):
    return (
      self._buildifier
      + self._clang_format
      + self._python
      + self._sqlformat
      + self._jsbeautify
      + self._gofmt
      + self._java
      + self._json
    )

  @property
  def modified_paths(self):
    return self._modified_paths

  def RunOrDie(self):
    linter_threads = []

    if self._buildifier:
      linter_threads.append(
        BuildiferThread(self._buildifier, verbose=self._verbose)
      )
    if self._clang_format:
      linter_threads.append(
        ClangFormatThread(self._clang_format, verbose=self._verbose)
      )
    if self._python:
      linter_threads.append(PythonThread(self._python, verbose=self._verbose))
    if self._sqlformat:
      linter_threads.append(SqlFormat(self._sqlformat, verbose=self._verbose))
    if self._jsbeautify:
      linter_threads.append(
        JsBeautifyThread(self._jsbeautify, verbose=self._verbose)
      )
    if self._gofmt:
      linter_threads.append(GoFmtThread(self._gofmt, verbose=self._verbose))
    if self._java:
      linter_threads.append(
        GoogleJavaFormatThread(self._java, verbose=self._verbose)
      )
    if self._json:
      linter_threads.append(JsonlintThread(self._json, verbose=self._verbose))

    for thread in linter_threads:
      thread.start()

    for thread in linter_threads:
      thread.join()
      self._modified_paths += thread.modified_paths


def GetGitDiffFilesOrDie(staged):
  """List *either* the staged or unstaged files (not both).

  To list both, call this function twice with both staged=True and staged=False.
  """
  cmd = ["git", "diff", "--name-only"]
  if staged:
    cmd.append("--cached")
  output = ExecOrDie(cmd)
  lines = output.split("\n")
  staged_file_relpaths = lines[:-1]  # Last line is blank.
  return staged_file_relpaths
