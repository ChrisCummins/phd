# Copyright (c) 2019-2020 Chris Cummins <chrisc.101@gmail.com>.
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
"""Utilities for working with bazel"""
import multiprocessing
import pathlib
import subprocess
import typing

from experimental.util.alice import alice_pb2
from labm8.py import app
from labm8.py import fs

FLAGS = app.FLAGS


class RunContext(typing.NamedTuple):
  process: typing.Any


class BazelError(OSError):
  pass


class BazelRunProcess(multiprocessing.Process):
  def __init__(
    self,
    run_request: alice_pb2.RunRequest,
    bazel_repo_dir: pathlib.Path,
    workdir: pathlib.Path,
  ):
    self.id = run_request.ledger_id
    self.bazel_repo_dir = bazel_repo_dir
    self.workdir = workdir
    super(BazelRunProcess, self).__init__(
      target=_BazelRunRequest, args=(run_request, bazel_repo_dir, workdir)
    )

  @property
  def pid(self) -> int:
    with open(self.workdir / "pid.txt") as f:
      return int(f.read())

  @property
  def returncode(self) -> typing.Optional[int]:
    if (self.workdir / "returncode.txt").is_file():
      with open(self.workdir / "returncode.txt") as f:
        return int(f.read())
    else:
      return None

  @property
  def stdout(self) -> str:
    with open(self.workdir / "stdout.txt") as f:
      return f.read()

  @property
  def stderr(self) -> str:
    with open(self.workdir / "stderr.txt") as f:
      return f.read()


class BazelClient(object):
  def __init__(self, root_dir: pathlib.Path, workdir: pathlib.Path):
    if not root_dir.is_dir():
      raise FileNotFoundError(f"Directory not found: {root_dir}")
    if not (root_dir / "WORKSPACE").is_file():
      raise BazelError(f"Workspace not found: {root_dir}/WORKSPACE")

    self._root_dir = root_dir
    self._workdir = workdir

  @property
  def root_dir(self) -> pathlib.Path:
    return self._root_dir

  @property
  def workdir(self) -> pathlib.Path:
    return self._workdir

  def Run(self, run_request: alice_pb2.RunRequest) -> BazelRunProcess:
    assert run_request.ledger_id
    workdir = self.workdir / str(run_request.ledger_id)
    workdir.mkdir()
    process = BazelRunProcess(run_request, self.root_dir, workdir)
    process.start()
    return process


def _BazelRunRequest(
  run_request: alice_pb2.RunRequest,
  bazel_repo_dir: pathlib.Path,
  workdir: pathlib.Path,
) -> None:
  cmd = (
    ["bazel", "run", run_request.target]
    + list(run_request.bazel_args)
    + ["--"]
    + list(run_request.bin_args)
  )
  if run_request.timeout_seconds:
    cmd = ["timeout", "-s9", str(run_request.timeout_seconds)] + cmd

  app.Log(1, "$ %s", " ".join(cmd))
  with fs.chdir(bazel_repo_dir):
    process = subprocess.Popen(
      cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

  stdout = subprocess.Popen(
    ["tee", str(workdir / "stdout.txt")], stdin=process.stdout
  )
  stderr = subprocess.Popen(
    ["tee", str(workdir / "stderr.txt")], stdin=process.stderr
  )

  # Record the ID of the current process.
  with open(workdir / "pid.txt", "w") as f:
    f.write(str(process.pid))

  process.communicate()
  stdout.communicate()
  stderr.communicate()
  returncode = process.returncode
  app.Log(1, "Process completed with returncode %d", returncode)
  with open(workdir / f"returncode.txt", "w") as f:
    f.write(str(returncode))
