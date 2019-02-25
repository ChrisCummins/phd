#!/usr/bin/env python
# Wrapper around bazel which checks that request bazel targets were included
# in source tree export.
from __future__ import print_function

import os
import subprocess
import sys

# The subset of bazel targets that are supported in this WORKSPACE.
EXPORTED_TARGETS = [
    # @EXPORTED_TARGETS@ #
]


def main():
  if not os.path.isfile('config.pbtxt'):
    print(
        '[bazel_wrapper.py] fatal: config.pbtxt not found. '
        'Must run ./configure first',
        file=sys.stderr)
    sys.exit(1)

  # For each argument, attempt to determine if it is a bazel target, and if it
  # is check that it was exported.
  for arg in sys.argv[1:]:
    if arg[0] != '-' and ('/' in arg or ':' in arg):
      if not arg.startswith('//'):
        arg = '//' + arg
      if arg not in EXPORTED_TARGETS:
        print(
            "Target {} not available. Available targets:\n  {}".format(
                arg, '\n  '.join(EXPORTED_TARGETS)),
            file=sys.stderr)
        sys.exit(1)
  print('[bazel_wrapper.py] Args are safe', file=sys.stderr)

  process = subprocess.Popen(['bazel'] + sys.argv[1:])
  process.communicate()
  sys.exit(process.returncode)


if __name__ == '__main__':
  main()
