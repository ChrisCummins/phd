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
import subprocess
import sys

# The path to the root of the PhD repository, i.e. the directory which this file
# is in.
# WARNING: Moving this file may require updating this path!
_PHD_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..')

# Ad-hoc import of //tools/code_style/linters:linters_lib.py
LINTERS_LIB = os.path.join(_PHD_ROOT, 'tools/code_style/linters/linters_lib.py')
if sys.version_info >= (3, 5):
  import importlib

  spec = importlib.util.spec_from_file_location("linters_lib", LINTERS_LIB)
  linters_lib = importlib.util.module_from_spec(spec)
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


def main(argv):
  assert not argv

  os.chdir(_PHD_ROOT)

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
    linters_lib.Print(commits_behind_upstream, 'commits behind')
    PullAndRebaseOrDie()
  else:
    linters_lib.Print('ok')

  files_that_exist = [f for f in staged_files if os.path.isfile(f)]
  linters = linters_lib.LinterActions(files_that_exist)
  num_actions = len(linters.paths_with_actions)

  if num_actions:
    linters_lib.Print('Running', num_actions, 'linter actions ...', end=' ')
    linters.RunOrDie()
    linters_lib.Print('ok')

    # Get a list of partially-staged files that were modified by the linters.
    partially_staged_files = set(
        f for f in linters.modified_paths if f in unstaged_files)

    # For every file in the git index where the entire diff was indexed,
    # re-index it.
    fully_staged_modified_files = [
        f for f in linters.modified_paths if f not in partially_staged_files
    ]

    if fully_staged_modified_files:
      linters_lib.Print('Modified files that will be automatically committed:')
      for path in sorted(fully_staged_modified_files):
        linters_lib.Print('   ', path)
      GitAddOrDie(fully_staged_modified_files)

    # Partially-staged files (i.e. the result of running `git add -p` cannot
    # be simply re-indexed and committed since we don't know what was modified
    # by the linter and what was deliberately left unstaged.
    if partially_staged_files:
      linters_lib.Print(
          'Partially staged modified files that must be inspected:')
      for path in sorted(list(partially_staged_files)):
        linters_lib.Print('   ', path)
      linters_lib.Print()
      linters_lib.Print('[action] Selectively add unstaged changes using:')
      linters_lib.Print()
      linters_lib.Print('    git add --patch --', *partially_staged_files)
      linters_lib.Print()
      sys.exit(1)

    linters_lib.Print('Pre-commit checks passed')


if __name__ == '__main__':
  try:
    main(sys.argv[1:])
  except KeyboardInterrupt:
    print('\ninterrupt', file=sys.stderr)
