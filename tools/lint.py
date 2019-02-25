#!/usr/bin/env python
"""Run linters on source files in this repository.

Usage:
  $ ./tools/lint.py [--all]

By default, only modified files will be linted. If --all argument is provided,
all git tracked files will be linted.
"""
from __future__ import print_function

import argparse
import os
import sys
import time

# The path to the root of the PhD repository, i.e. the directory which this file
# is in.
# WARNING: Moving this file may require updating this path!
_PHD_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..')
assert os.path.isdir(os.path.join(_PHD_ROOT, '.git'))

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


def main(argv):
  os.chdir(_PHD_ROOT)

  parser = argparse.ArgumentParser()
  parser.add_argument('--all', action='store_true')

  args = parser.parse_args(argv)

  if args.all:
    lint_candidates = linters_lib.ExecOrDie(['git',
                                             'ls-files']).rstrip().split('\n')
  else:
    lint_candidates = set(
        linters_lib.GetGitDiffFilesOrDie(staged=False) +
        linters_lib.GetGitDiffFilesOrDie(staged=True))

  # Changed files may include deleted files.
  files_that_exist = [f for f in lint_candidates if os.path.isfile(f)]

  linters = linters_lib.LinterActions(files_that_exist)

  start_time = time.time()
  linters_lib.Print(
      'Performing',
      len(linters.paths_with_actions),
      'lint actions ...',
      end=' ')
  linters.RunOrDie()
  print('{:.3f}s'.format(time.time() - start_time))

  for path in sorted(linters.modified_paths):
    print('   ', path)


if __name__ == '__main__':
  try:
    main(sys.argv[1:])
  except KeyboardInterrupt:
    print('\ninterrupt', file=sys.stderr)
