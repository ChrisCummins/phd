#!/usr/bin/env python
"""Pre-commit git hook.

This script performs a handful of actions:
  * It checks to see if there are any new commits on the remote.
  * If there are, it rebases the staged changes atop them.
  * It runs the linters on the changed files. When possible, linter
    modifications are automatically committed.
"""
from __future__ import print_function

import os
import sys
import time

import subprocess

# The path to the root of the PhD repository, i.e. the directory which this file
# is in.
# WARNING: Moving this file may require updating this path!
_PHD_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')

# Ad-hoc import of //tools/code_style/linters:linters_lib.py
LINTERS_LIB = os.path.join(_PHD_ROOT, 'tools/code_style/linters/linters_lib.py')
if sys.version_info >= (3, 5):
  from importlib import util

  spec = util.spec_from_file_location("linters_lib", LINTERS_LIB)
  linters_lib = util.module_from_spec(spec)
  spec.loader.exec_module(linters_lib)
else:
  import imp

  linters_lib = imp.load_source('linters_lib', LINTERS_LIB)


def GetCommitsBehindUpstreamOrDie(remote_name, branch_name):
  linters_lib.ExecOrDie(['git', 'fetch', remote_name])
  outputs = subprocess.check_output([
      'git', 'rev-list', '--left-right', '--count',
      remote_name + '/' + branch_name + '...@'
  ],
                                    universal_newlines=True)
  commits_behind_upstream = outputs.split()[0]
  return int(commits_behind_upstream)


def PullAndRebaseOrDie():
  """Update to meet"""
  # Create stash.
  linters_lib.ExecOrDie(['git', 'stash', 'save', ''])

  # Rebase.
  linters_lib.ExecOrDie(['git', 'pull', '--rebase'])

  # Apply the stash and reset the index.
  try:
    subprocess.check_call(['git', 'stash', 'pop', '--index'],
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE)
  except subprocess.CalledProcessError:
    # If the stash fails to pop, this is because of a merge conflict. Pop the
    # stash but without restoring the conflicted index. This command will
    # cause the script to abort.
    linters_lib.ExecOrDie(['git', 'stash', 'pop'])


def GitAddOrDie(paths):
  linters_lib.ExecOrDie(['git', 'add'] + paths)


def GetMd5sumOutputAsString(paths):
  """Return the output of md5sum on a list of paths as a string."""
  if not paths:
    return ''
  return subprocess.check_output(
      ['timeout', '-s9', '60', 'md5sum'] + paths, universal_newlines=True)


class Gazelle(object):
  """Utilities for running gazelle.

  Gazelle is a bazel build file generator for go.

  See: https://github.com/bazelbuild/bazel-gazelle
  """

  @classmethod
  def RunGazelle(cls):
    """Run gazelle, a tool"""
    subprocess.check_call(['bazel', 'run', '//:gazelle'])
    cls._RenameBuildFiles()

  @classmethod
  def MaybeRunGazelle(cls, paths):
    """Determine if gazelle should be run, and run it if so.

    Args:
      paths: A list of paths modified in this commit.
    """
    if cls._ListContainsFilesModifiedByGazelle(paths):
      cls.RunGazelle()

  @staticmethod
  def _ListContainsFilesModifiedByGazelle(paths):
    """Determine if any of the files in the list are interesting to Gazelle.

    Gazelle should only care about golang files, proto libraries (because it
    generates proto rules for them), and BUILD files (because they may contain
    go rules).

    Args:
      paths: A list of paths.
    """
    for path in paths:
      if path.endswith('.go') or path.endswith('.proto'):
        return True
    return False

  @staticmethod
  def _RenameBuildFiles():
    find_output = subprocess.check_output(['find', '.', '-name', 'BUILD.bazel'],
                                          universal_newlines=True).rstrip()
    if not find_output:
      return
    files_to_rename = find_output.split('\n')
    for path in files_to_rename:
      new_path = path[:-len(".bazel")]
      print("Renaming {} -> {}".format(path, new_path))
      os.rename(path, new_path)


def main(argv):
  assert not argv

  start_time = time.time()

  os.chdir(_PHD_ROOT)

  task_start_time = time.time()
  branch_name = linters_lib.GetGitBranchOrDie()
  remote_name = linters_lib.GetGitRemoteOrDie(branch_name)
  staged_files = linters_lib.GetGitDiffFilesOrDie(staged=True)
  unstaged_files = linters_lib.GetGitDiffFilesOrDie(staged=False)

  linters_lib.Print(
      'Checking if',
      branch_name,
      'is up to date with',
      remote_name,
      '...',
      end=' ')
  commits_behind_upstream = GetCommitsBehindUpstreamOrDie(
      remote_name, branch_name)
  if commits_behind_upstream:
    linters_lib.Print('⚠️  ', commits_behind_upstream, 'commits behind')
    PullAndRebaseOrDie()
  else:
    linters_lib.Print('ok  {:.3f}s'.format(time.time() - task_start_time))

  files_that_exist = [f for f in staged_files if os.path.isfile(f)]

  # A list of files that will be automatically staged for commit.
  files_to_stage = []
  # A list of files that were partially staged and have been modified by this
  # script. These must be hand-inspected and will cause the commit to abort.
  partially_staged_modified_files = []

  # Files that are both staged and unstaged are partially staged.
  partially_staged_files = list(
      set(staged_files).intersection(set(unstaged_files)))
  # We're only interest in files that exist.
  partially_staged_files = [
      f for f in partially_staged_files if os.path.isfile(f)
  ]

  partially_staged_checksums = GetMd5sumOutputAsString(partially_staged_files)

  # Run gazelle before linters.
  task_start_time = time.time()
  linters_lib.Print('Running gazelle ...', end=' ')
  Gazelle.MaybeRunGazelle(files_that_exist)

  # Check to see if gazelle modified any partially-staged files.
  new_partially_staged_checksums = GetMd5sumOutputAsString(
      partially_staged_files)
  if new_partially_staged_checksums != partially_staged_checksums:
    # Go line by line through the md5sum outputs to find the differing files.
    for left, right in zip(
        partially_staged_checksums.split('\n'),
        new_partially_staged_checksums.split('\n')):
      if left != right:
        partially_staged_modified_files.append(' '.join(left.split()[1:]))
  linters_lib.Print('ok  {:.3f}s'.format(time.time() - task_start_time))

  linters = linters_lib.LinterActions(files_that_exist)
  num_actions = len(linters.paths_with_actions)

  if num_actions:
    task_start_time = time.time()
    linters_lib.Print('Running', num_actions, 'linter actions ...', end=' ')
    linters.RunOrDie()
    linters_lib.Print('ok  {:.3f}s'.format(time.time() - task_start_time))

    # Get a list of partially-staged files that were modified by the linters.
    partially_staged_files = set(
        f for f in linters.modified_paths if f in unstaged_files)
    partially_staged_modified_files += list(partially_staged_files)

    # For every file in the git index where the entire diff was indexed,
    # re-index it.
    files_to_stage += [
        f for f in linters.modified_paths if f not in partially_staged_files
    ]

    if files_to_stage:
      files_to_stage = list(set(files_to_stage))
      linters_lib.Print('Modified files that will be automatically committed:')
      for path in sorted(files_to_stage):
        linters_lib.Print('   ', path)
      GitAddOrDie(files_to_stage)

    # Partially-staged files (i.e. the result of running `git add -p` cannot
    # be simply re-indexed and committed since we don't know what was modified
    # by the linter and what was deliberately left unstaged.
    if partially_staged_modified_files:
      partially_staged_modified_files = list(
          sorted(set(partially_staged_modified_files)))
      linters_lib.Print(
          '⚠️  Partially staged modified files that must be inspected:')
      for path in partially_staged_modified_files:
        linters_lib.Print('   ', path)
      linters_lib.Print()
      linters_lib.Print('[action] Selectively add unstaged changes using:')
      linters_lib.Print()
      linters_lib.Print('    git add --patch --',
                        *partially_staged_modified_files)
      linters_lib.Print()
      sys.exit(1)

    linters_lib.Print(
        '✅  Pre-commit checks passed in {:.3f}s'.format(time.time() -
                                                        start_time))


if __name__ == '__main__':
  try:
    main(sys.argv[1:])
  except KeyboardInterrupt:
    print('\ninterrupt', file=sys.stderr)
