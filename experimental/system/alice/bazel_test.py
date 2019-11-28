# Copyright (c) 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# alice is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# alice is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with alice.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //experimental/system/alice:bazel.py."""
import os
import pathlib
import signal
import time

import pytest

from experimental.system.alice import alice_pb2
from experimental.system.alice import bazel
from labm8.py import system
from labm8.py import test

DUMMY_TARGET = "//experimental/system/alice/test:dummy_target"


@pytest.fixture(scope="module")
def workspace(module_tempdir: pathlib.Path) -> pathlib.Path:
  """Create a workspace with a single //:hello binary target."""
  with open(module_tempdir / "WORKSPACE", "w") as f:
    f.write(
      """
workspace(name = "test")
"""
    )

  with open(module_tempdir / "hello.cc", "w") as f:
    f.write(
      """
#include <iostream>

int main(int argc, char** argv) {
  std::cout << "Hello, stdout!\\n";
  std::cerr << "Hello, stderr!\\n";
  return 0;
}
"""
    )

  with open(module_tempdir / "BUILD", "w") as f:
    f.write(
      """
cc_binary(
    name = "hello",
    srcs = ["hello.cc"],
)
"""
    )

  yield module_tempdir


def test_BazelClient_root_dir_not_found(tempdir: pathlib.Path):
  """Error is raised if root dir is not found."""
  with pytest.raises(FileNotFoundError):
    bazel.BazelClient(tempdir / "foo", tempdir / "work")


def test_BazelClient_workspace_not_found(tempdir: pathlib.Path):
  """Error is raised if WORKSPACE is not found."""
  with pytest.raises(bazel.BazelError):
    (tempdir / "repo").mkdir()
    bazel.BazelClient(tempdir / "repo", tempdir / "work")


def test_BazelClient_Run_returncode(
  workspace: pathlib.Path, tempdir: pathlib.Path
):
  """Check returncode of test target."""
  client = bazel.BazelClient(workspace, tempdir)
  process = client.Run(alice_pb2.RunRequest(target="//:hello", ledger_id=1,))

  process.join()
  assert process.returncode == 0


def test_BazelClient_Run_stdout(workspace: pathlib.Path, tempdir: pathlib.Path):
  """Check stdout of test target."""
  client = bazel.BazelClient(workspace, tempdir)
  process = client.Run(alice_pb2.RunRequest(target="//:hello", ledger_id=1,))

  process.join()
  assert process.stdout == "Hello, stdout!\n"


def test_BazelClient_Run_process_id(
  workspace: pathlib.Path, tempdir: pathlib.Path
):
  """Check that process ID is set."""
  client = bazel.BazelClient(workspace, tempdir)
  process = client.Run(alice_pb2.RunRequest(target="//:hello", ledger_id=1,))

  process.join()
  assert process.pid
  assert process.pid != system.PID


@pytest.mark.xfail(reason="FIXME")
def test_BazelClient_Run_stderr(workspace: pathlib.Path, tempdir: pathlib.Path):
  """Check stderr of test target."""
  client = bazel.BazelClient(workspace, tempdir)
  process = client.Run(alice_pb2.RunRequest(target="//:hello", ledger_id=1,))

  process.join()
  # Stderr starts with bazel build log.
  assert process.stderr.endswith("Hello, stderr!\n")


def test_BazelClient_Run_workdir_files(
  workspace: pathlib.Path, tempdir: pathlib.Path
):
  """Check that output files are generated."""
  client = bazel.BazelClient(workspace, tempdir)
  process = client.Run(alice_pb2.RunRequest(target="//:hello", ledger_id=1,))

  process.join()
  assert (process.workdir / "stdout.txt").is_file()
  assert (process.workdir / "stderr.txt").is_file()
  assert (process.workdir / "returncode.txt").is_file()


def test_BazeClient_run_missing_target(
  workspace: pathlib.Path, tempdir: pathlib.Path
):
  """Check error for missing target."""
  client = bazel.BazelClient(workspace, tempdir)
  process = client.Run(
    alice_pb2.RunRequest(target="//:not_a_target", ledger_id=1,)
  )

  process.join()
  assert process.returncode


def test_BazelClient_Run_process_isnt_running(
  workspace: pathlib.Path, tempdir: pathlib.Path
):
  """Check that process isn't running after completed."""
  client = bazel.BazelClient(workspace, tempdir)
  process = client.Run(alice_pb2.RunRequest(target="//:hello", ledger_id=1,))

  process.join()
  try:
    os.kill(process.pid, 0)
    pytest.fail(
      "os.kill() didn't fail, that means the process is still " "running"
    )
  except ProcessLookupError:
    pass


def test_BazelClient_kill_process(
  workspace: pathlib.Path, tempdir: pathlib.Path
):
  """Test that process can be killed."""
  # Create a binary which will never terminate once launched.
  with open(workspace / "BUILD", "a") as f:
    f.write(
      """
cc_binary(
    name = "nonterminating",
    srcs = ["nonterminating.cc"],
)
"""
    )

  with open(workspace / "nonterminating.cc", "w") as f:
    f.write(
      """
int main() {
  while (1) {}
}
"""
    )

  client = bazel.BazelClient(workspace, tempdir)
  process = client.Run(
    alice_pb2.RunRequest(target="//:nonterminating", ledger_id=1,)
  )

  # Sleep for luck ;-)
  time.sleep(3)

  # Send the non-terminating process a kill signal.
  os.kill(process.pid, signal.SIGTERM)
  process.join()
  try:
    os.kill(process.pid, 0)
    pytest.fail(
      "os.kill() didn't fail, that means the process is still " "running"
    )
  except ProcessLookupError:
    pass


if __name__ == "__main__":
  test.Main()
