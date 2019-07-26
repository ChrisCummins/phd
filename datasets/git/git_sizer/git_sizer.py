"""Wrapper around git-sizer.

Compute various size metrics for a Git repository, flagging those that might
cause problems.

https://github.com/github/git-sizer/
"""

import pathlib
import subprocess

from labm8 import app
from labm8 import bazelutil
from labm8 import fs
from labm8 import system

FLAGS = app.FLAGS

_GIT_SIZER_REPO = 'git_sizer_linux' if system.is_linux() else 'git_sizer_mac'
_GIT_SIZER = bazelutil.DataPath(f'{_GIT_SIZER_REPO}/git-sizer')

app.DEFINE_input_path('path', '.', 'The path to run git-sizer in.', is_dir=True)


def RunGitSizer(repo_root: pathlib.Path, *args, timeout_seconds: int = 360):
  """Run git-sizer binary in the given directory, with additional args."""
  with fs.chdir(repo_root):
    cmd = (['timeout', '-s9',
            str(timeout_seconds),
            str(_GIT_SIZER)] + list(args))
    app.Log(2, '$ %s', ' '.join(cmd))
    subprocess.check_call(cmd)


def main(argv):
  """Main entry point."""
  try:
    RunGitSizer(FLAGS.path, *argv[1:])
  except subprocess.CalledProcessError as e:
    app.FatalWithoutStackTrace(str(e))


if __name__ == '__main__':
  app.RunWithArgs(main)
