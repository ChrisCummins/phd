"""Unit tests for //alice:bazel.py."""
import pathlib

import pytest

from alice import alice_pb2
from alice import bazel
from labm8 import test


DUMMY_TARGET = '//alice/test:dummy_target'


@pytest.fixture(scope='module')
def workspace(module_tempdir: pathlib.Path) -> pathlib.Path:
  """Create a workspace with a single //:hello binary target."""
  with open(module_tempdir / 'WORKSPACE', 'w') as f:
    f.write("""
workspace(name = "test")
""")

  with open(module_tempdir / 'hello.cc', 'w') as f:
    f.write("""
#include <iostream>

int main(int argc, char** argv) {
  std::cout << "Hello, stdout!\\n";
  std::cerr << "Hello, stderr!\\n";
  return 0;
}
""")

  with open(module_tempdir / 'BUILD', 'w') as f:
    f.write("""
cc_binary(
    name = "hello",
    srcs = ["hello.cc"],
)
""")

  yield module_tempdir


def test_BazelClient_root_dir_not_found(tempdir: pathlib.Path):
  """Error is raised if root dir is not found."""
  with pytest.raises(FileNotFoundError):
    bazel.BazelClient(tempdir / 'foo', tempdir / 'work')


def test_BazelClient_workspace_not_found(tempdir: pathlib.Path):
  """Error is raised if WORKSPACE is not found."""
  with pytest.raises(bazel.BazelError):
    (tempdir / 'repo').mkdir()
    bazel.BazelClient(tempdir / 'repo', tempdir / 'work')


def test_BazelClient_Run_returncode(
    workspace: pathlib.Path, tempdir: pathlib.Path):
  """Check returncode of test target."""
  client = bazel.BazelClient(workspace, tempdir)
  process = client.Run(alice_pb2.RunRequest(
      target='//:hello',
      bazel_args=[],
      bin_args=[],
      ledger_id=1,
  ))

  process.join()
  assert process.returncode == 0


def test_BazelClient_Run_stdout(
    workspace: pathlib.Path, tempdir: pathlib.Path):
  """Check stdout of test target."""
  client = bazel.BazelClient(workspace, tempdir)
  process = client.Run(alice_pb2.RunRequest(
      target='//:hello',
      bazel_args=[],
      bin_args=[],
      ledger_id=1,
  ))

  process.join()
  assert process.stdout == 'Hello, stdout!\n'


@pytest.mark.xfail(reason='FIXME')
def test_BazelClient_Run_stderr(
    workspace: pathlib.Path, tempdir: pathlib.Path):
  """Check stderr of test target."""
  client = bazel.BazelClient(workspace, tempdir)
  process = client.Run(alice_pb2.RunRequest(
      target='//:hello',
      bazel_args=[],
      bin_args=[],
      ledger_id=1,
  ))

  process.join()
  # Stderr starts with bazel build log.
  assert process.stderr.endswith('Hello, stderr!\n')


def test_BazelClient_Run_workdir_files(
    workspace: pathlib.Path, tempdir: pathlib.Path):
  """Check that output files are generated."""
  client = bazel.BazelClient(workspace, tempdir)
  process = client.Run(alice_pb2.RunRequest(
      target='//:hello',
      bazel_args=[],
      bin_args=[],
      ledger_id=1,
  ))

  process.join()
  assert (process.workdir / 'stdout.txt').is_file()
  assert (process.workdir / 'stderr.txt').is_file()
  assert (process.workdir / 'returncode.txt').is_file()


def test_BazeClient_run_missing_target(
    workspace: pathlib.Path, tempdir: pathlib.Path):
  """Check error for missing target."""
  client = bazel.BazelClient(workspace, tempdir)
  process = client.Run(alice_pb2.RunRequest(
      target='//:not_a_target',
      bazel_args=[],
      bin_args=[],
      ledger_id=1,
  ))

  process.join()
  assert process.returncode


if __name__ == '__main__':
  test.Main()
