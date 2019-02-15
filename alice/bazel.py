"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
import collections
import logging
import multiprocessing
import pathlib
import subprocess
import sys
import typing

from absl import flags

from alice import alice_pb2
from labm8 import fs


FLAGS = flags.FLAGS

RunContext = collections.namedtuple('RunContext', ['process'])


class BazelError(OSError):
  pass


class BazelRunProcess(multiprocessing.Process):

  def __init__(self, run_request: alice_pb2.RunRequest,
               bazel_repo_dir: pathlib.Path,
               workdir: pathlib.Path):
    self.id = run_request.ledger_id
    self.bazel_repo_dir = bazel_repo_dir
    self.workdir = workdir
    super(BazelRunProcess, self).__init__(
        target=_BazelRunRequest,
        args=(run_request, bazel_repo_dir, workdir))

  @property
  def pid(self) -> int:
    with open(self.workdir / 'pid.txt') as f:
      return int(f.read())

  @property
  def returncode(self) -> typing.Optional[int]:
    if (self.workdir / 'returncode.txt').is_file():
      with open(self.workdir / 'returncode.txt') as f:
        return int(f.read())
    else:
      return None

  @property
  def stdout(self) -> str:
    with open(self.workdir / 'stdout.txt') as f:
      return f.read()

  @property
  def stderr(self) -> str:
    with open(self.workdir / 'stderr.txt') as f:
      return f.read()


class BazelClient(object):

  def __init__(self, root_dir: pathlib.Path,
               workdir: pathlib.Path):
    if not root_dir.is_dir():
      raise FileNotFoundError(f"Directory not found: {root_dir}")
    if not (root_dir / 'WORKSPACE').is_file():
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


def _BazelRunRequest(run_request: alice_pb2.RunRequest,
                     bazel_repo_dir: pathlib.Path,
                     workdir: pathlib.Path) -> None:
  cmd = (['bazel', 'run', run_request.target] +
         list(run_request.bazel_args) + ['--'] +
         list(run_request.bin_args))
  if run_request.timeout_seconds:
    cmd = ['timeout', '-s9', str(run_request.timeout_seconds)] + cmd

  logging.info('$ %s', ' '.join(cmd))
  with fs.chdir(bazel_repo_dir):
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)


  stdout = subprocess.Popen(['tee', str(workdir / 'stdout.txt')],
                            stdin=process.stdout)
  stderr = subprocess.Popen(['tee', str(workdir / 'stderr.txt')],
                            stdin=process.stderr)

  # Record the ID of the current process.
  with open(workdir / 'pid.txt', 'w') as f:
    f.write(str(process.pid))

  process.communicate()
  stdout.communicate()
  stderr.communicate()
  returncode = process.returncode
  logging.info("Process completed with returncode %d", returncode)
  with open(workdir / f'returncode.txt', 'w') as f:
    f.write(str(returncode))
