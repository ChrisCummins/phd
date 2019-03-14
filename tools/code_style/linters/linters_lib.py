"""A standalone module of utility code for linting.

Code in this file can have no phd dependencies, and is invoked directly, not
through bazel.
"""
from __future__ import print_function

import os
import subprocess
import sys
import threading

# The path to the root of the PhD repository, i.e. the directory which this file
# is in.
# WARNING: Moving this file may require updating this path!
_PHD_ROOT = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '../../..')


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
YAPF = WhichOrDie('yapf')
SQLFORMAT = WhichOrDie('sqlformat')
JSBEAUTIFY = WhichOrDie('js-beautify')

YAPF_RC = os.path.join(_PHD_ROOT, 'tools/code_style/yapf.yml')
assert os.path.isfile(YAPF_RC)
JSBEAUTIFY_RC = os.path.join(_PHD_ROOT, 'tools/code_style/jsbeautifyrc.json')
assert os.path.isfile(JSBEAUTIFY_RC)


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
      cmd,
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

  def __init__(self, paths):
    super(BuildiferThread, self).__init__(paths)

  def run(self):
    ExecOrDie([BUILDIFIER] + self._paths)


class ClangFormatThread(LinterThread):

  def __init__(self, paths):
    super(ClangFormatThread, self).__init__(paths)

  def run(self):
    # TODO(cec): Use project-local clang-format style file.
    ExecOrDie([CLANG_FORMAT, '-style', 'Google', '-i'] + self._paths)


class YapfThread(LinterThread):

  def __init__(self, paths):
    super(YapfThread, self).__init__(paths)

  def run(self):
    ExecOrDie([YAPF, '--style', YAPF_RC, '-i'] + self._paths)


class SqlFormatThread(LinterThread):

  def __init__(self, paths):
    super(SqlFormatThread, self).__init__(paths)

  def run(self):
    for path in self.paths:
      ExecOrDie([
          SQLFORMAT, '--reindent', '--keywords', 'upper', '--identifiers',
          'lower', path, '--outfile', path
      ])


class JsBeautifyThread(LinterThread):

  def run(self):
    ExecOrDie([JSBEAUTIFY, '--config', JSBEAUTIFY_RC] + self._paths)


class LinterActions(object):

  def __init__(self, paths):
    self._paths = paths
    self._buildifier = []
    self._clang_format = []
    self._yapf = []
    self._sqlformat = []
    self._jsbeautify = []
    self._modified_paths = []

    for path in paths:
      basename = os.path.basename(path)
      if basename == 'BUILD' or basename == 'WORKSPACE':
        self._buildifier.append(path)

      _, extension = os.path.splitext(path)

      if extension == '.cc' or extension == '.c' or extension == '.h':
        self._clang_format.append(path)
      elif extension == '.py':
        self._yapf.append(path)
      elif extension == '.sql':
        self._sqlformat.append(path)
      elif (extension == '.html' or extension == '.css' or
            extension == '.scss' or extension == '.js'):
        self._jsbeautify.append(path)

  @property
  def paths(self):
    return self._paths

  @property
  def paths_with_actions(self):
    return (self._buildifier + self._clang_format + self._yapf + self._sqlformat
            + self._jsbeautify)

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
      linter_threads.append(SqlFormatThread(self._sqlformat))
    if self._jsbeautify:
      linter_threads.append(JsBeautifyThread(self._jsbeautify))

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
