"""A standalone module of utility code for linting.

Code in this file can have no phd dependencies, and is invoked directly, not
through bazel.
"""
from __future__ import print_function

import os
import sys

import subprocess
import threading

# The path to the root of the PhD repository, i.e. the directory which this file
# is in.
# WARNING: Moving this file may require updating this path!
_PHD_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                         '../../..')


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
  print('fatal: Could not find required binary:', name, file=sys.stderr)
  sys.exit(1)


BUILDIFIER = WhichOrDie('buildifier')
CLANG_FORMAT = WhichOrDie('clang-format')
GO = WhichOrDie('go')
JAVA = WhichOrDie('java')
JSBEAUTIFY = WhichOrDie('js-beautify')
JSON_LINT = WhichOrDie('jsonlint')
SQLFORMAT = WhichOrDie('sqlformat')
YAPF = WhichOrDie('yapf')

YAPF_RC = os.path.join(_PHD_ROOT, 'tools/code_style/yapf.yml')
assert os.path.isfile(YAPF_RC)
JSBEAUTIFY_RC = os.path.join(_PHD_ROOT, 'tools/code_style/jsbeautifyrc.json')
assert os.path.isfile(JSBEAUTIFY_RC)
GOOGLE_JAVA_FORMAT = os.path.join(
    _PHD_ROOT, 'tools/code_style/linters/google-java-format-1.7-all-deps.jar')
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
  process = subprocess.Popen(cmd,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             universal_newlines=True)
  stdout, stderr = process.communicate()
  if process.returncode:
    Print('error')
    Print('    $ ', ' '.join(cmd), sep='')

    for line in stderr.rstrip().split('\n'):
      Print('   ', line)

    sys.exit(1)
  return stdout


def GetGitBranchOrDie():
  """Get the name of the current git branch."""
  branches = ExecOrDie(['git', 'branch'])
  for line in branches.split('\n'):
    if line.startswith('* '):
      return line[2:]
  print("fatal: Unable to determine git branch", file=sys.stderr)
  sys.exit(1)


def GetGitRemoteOrDie(branch_name):
  """Get the name of the current git remote."""
  fully_qualified_name = ExecOrDie(
      ['git', 'rev-parse', '--abbrev-ref', branch_name + '@{upstream}'])
  components = fully_qualified_name.split('/')
  assert len(components) > 1
  return components[0]


class LinterThread(threading.Thread):

  def __init__(self, paths):
    super(LinterThread, self).__init__()
    assert paths
    self._original_mtimes = [os.path.getmtime(f) for f in paths]
    self._paths = paths

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


class BuildiferThread(LinterThread):

  def run(self):
    ExecOrDie([BUILDIFIER] + self._paths)


class ClangFormatThread(LinterThread):

  def run(self):
    # TODO(cec): Use project-local clang-format style file.
    ExecOrDie([CLANG_FORMAT, '-style', 'Google', '-i'] + self._paths)


class YapfThread(LinterThread):

  def run(self):
    ExecOrDie([YAPF, '--style', YAPF_RC, '-i'] + self._paths)


class SqlFormat(LinterThread):

  def run(self):
    for path in self.paths:
      self.Lint(path)

  @staticmethod
  def Lint(path):
    ExecOrDie([
        SQLFORMAT, '--reindent', '--keywords', 'upper', '--identifiers',
        'lower', path, '--outfile', path
    ])


class JsBeautifyThread(LinterThread):

  def run(self):
    ExecOrDie([JSBEAUTIFY, '--config', JSBEAUTIFY_RC] + self._paths)


class GoFmtThread(LinterThread):

  def run(self):
    # Run linter on each file individually because:
    #   1. An error in one file prevents linting in all other files.
    #   2. All files in a single invocation must be in the same directory.
    for path in self._paths:
      ExecOrDie([GO, 'fmt', path])


class GoogleJavaFormatThread(LinterThread):

  def run(self):
    ExecOrDie([JAVA, '-jar', GOOGLE_JAVA_FORMAT, '-i'] + self._paths)


class JsonlintThread(LinterThread):

  def run(self):
    for path in self._paths:
      ExecOrDie([JSON_LINT, '-i', path])


class LinterActions(object):

  def __init__(self, paths):
    self._paths = paths
    self._buildifier = []
    self._clang_format = []
    self._yapf = []
    self._sqlformat = []
    self._jsbeautify = []
    self._gofmt = []
    self._java = []
    self._json = []
    self._modified_paths = []

    for path in paths:
      basename = os.path.basename(path)
      if basename == 'BUILD' or basename == 'WORKSPACE':
        self._buildifier.append(path)

      _, extension = os.path.splitext(path)

      if extension == '.cc' or extension == '.c' or extension == '.h' or extension == '.ino':
        self._clang_format.append(path)
      elif extension == '.py':
        self._yapf.append(path)
      elif extension == '.sql':
        self._sqlformat.append(path)
      elif extension == '.html' or extension == '.css' or extension == '.js':
        self._jsbeautify.append(path)
      elif extension == '.go':
        self._gofmt.append(path)
      elif extension == '.java':
        self._java.append(path)
      elif extension == '.json':
        self._json.append(path)

  @property
  def paths(self):
    return self._paths

  @property
  def paths_with_actions(self):
    return (self._buildifier + self._clang_format + self._yapf +
            self._sqlformat + self._jsbeautify + self._gofmt + self._java +
            self._json)

  @property
  def modified_paths(self):
    return self._modified_paths

  def RunOrDie(self):
    linter_threads = []

    if self._buildifier:
      linter_threads.append(BuildiferThread(self._buildifier))
    if self._clang_format:
      linter_threads.append(ClangFormatThread(self._clang_format))
    if self._yapf:
      linter_threads.append(YapfThread(self._yapf))
    if self._sqlformat:
      linter_threads.append(SqlFormat(self._sqlformat))
    if self._jsbeautify:
      linter_threads.append(JsBeautifyThread(self._jsbeautify))
    if self._gofmt:
      linter_threads.append(GoFmtThread(self._gofmt))
    if self._java:
      linter_threads.append(GoogleJavaFormatThread(self._java))
    if self._json:
      linter_threads.append(JsonlintThread(self._json))

    for thread in linter_threads:
      thread.start()

    for thread in linter_threads:
      thread.join()
      self._modified_paths += thread.modified_paths


def GetGitDiffFilesOrDie(staged):
  """List *either* the staged or unstaged files (not both).

  To list both, call this function twice with both staged=True and staged=False.
  """
  cmd = ['git', 'diff', '--name-only']
  if staged:
    cmd.append('--cached')
  output = ExecOrDie(cmd)
  lines = output.split('\n')
  staged_file_relpaths = lines[:-1]  # Last line is blank.
  return staged_file_relpaths
