#!/usr/bin/env python
"""Post-commit git hook.

This script pushes the freshly minted commit to remote.
"""
from __future__ import print_function

import os
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


def GitPushOrDie(branch_name, remote_name):
  """Push from current head to remote branch."""
  linters_lib.ExecOrDie(['git', 'push', remote_name, branch_name])


def main(argv):
  assert not argv
  os.chdir(_PHD_ROOT)

  branch_name = linters_lib.GetGitBranchOrDie()
  remote_name = linters_lib.GetGitRemoteOrDie(branch_name)

  linters_lib.Print('Pushing', branch_name, 'to', remote_name, '...', end=' ')
  GitPushOrDie(branch_name, remote_name)
  linters_lib.Print('ok')

  linters_lib.Print('Post-commit checks passed')


if __name__ == '__main__':
  try:
    main(sys.argv[1:])
  except KeyboardInterrupt:
    print('\ninterrupt', file=sys.stderr)
