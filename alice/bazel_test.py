"""Unit tests for //alice:bazel.py."""
import pathlib

import pytest

from alice import bazel
from labm8 import test


DUMMY_TARGET = '//alice/test:dummy_target'


@pytest.fixture(scope='function')
def workspace(tempdir: pathlib.Path) -> pathlib.Path:
  """Create a workspace with a single //:hello binary target."""
  with open(tempdir / 'WORKSPACE', 'w') as f:
    f.write("""
workspace(name = "test")
""")

  with open(tempdir / 'hello.cc', 'w') as f:
    f.write("""
#include <iostream>

int main(int argc, char** argv) {
  std::cout << "Hello, stdout!\\n";
  std::cerr << "Hello, stderr!\\n";
  return 0;
}
""")

  with open(tempdir / 'BUILD', 'w') as f:
    f.write("""
cc_binary(
    name = "hello",
    srcs = ["hello.cc"],
)
""")

  yield tempdir


def test_BazelClient_root_dir_not_found(tempdir: pathlib.Path):
  """Short summary of test."""
  with pytest.raises(FileNotFoundError):
    bazel.BazelClient(tempdir / 'foo')


def test_BazelClient_workspace_not_found(tempdir: pathlib.Path):
  """Short summary of test."""
  with pytest.raises(bazel.BazelError):
    bazel.BazelClient(tempdir)


def test_BazelClient_run_binary_target(workspace: pathlib.Path):
  """Short summary of test."""
  client = bazel.BazelClient(workspace)
  ctx = client.GetRunContext('//:hello', [], [])
  stdout, stderr = ctx.process.communicate()
  assert stdout.decode('utf-8') == 'Hello, stdout!\n'

  # Stderr will begin with the bazel build stuff.
  assert stderr.decode('utf-8').endswith('Hello, stderr!\n')


def test_BazeClient_run_missing_target(workspace: pathlib.Path):
  client = bazel.BazelClient(workspace)
  ctx = client.GetRunContext('//:not_a_target', [], [])

  stdout, stderr = ctx.process.communicate()


#
# def test_BazeClient_run_missing_target(workspace: pathlib.Path):
#   client = bazel.BazelClient(workspace)
#   client.GetRunContext('//:hello', ['-invalid', '-flag'], [])


if __name__ == '__main__':
  test.Main()
