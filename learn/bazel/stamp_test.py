"""Learning how build stamping works.

Stamping is the process by which bazel encodes build information into builds.

https://docs.bazel.build/versions/master/user-manual.html#workspace_status
"""

from labm8 import bazelutil
from labm8 import fs
from labm8 import test

STAMP_FILE = bazelutil.DataPath('phd/learn/bazel/stamp_file.txt')


def test_StampFile():
  stamp_file = fs.Read(STAMP_FILE)
  print(stamp_file)
  assert 'STABLE_BUILD_GIT_HASH' in stamp_file


if __name__ == '__main__':
  test.Main()
