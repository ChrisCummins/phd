"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
import collections
import logging
import multiprocessing
import pathlib
import subprocess
import sys

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


class BazelClient(object):

  def __init__(self, root_dir: pathlib.Path):
    if not root_dir.is_dir():
      raise FileNotFoundError(f"Directory not found: {root_dir}")
    if not (root_dir / 'WORKSPACE').is_file():
      raise BazelError(f"Workspace not found: {root_dir}/WORKSPACE")

    self._root_dir = root_dir

  @property
  def root_dir(self) -> pathlib.Path:
    return self._root_dir

  def Run(self, run_request: alice_pb2.RunRequest,
          workdir: pathlib.Path) -> BazelRunProcess:
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

  with open(workdir / 'stdout.txt', 'wb') as stdout_log:
    with open(workdir / 'stderr.txt', 'wb') as stderr_log:
      while process.poll() is None:
        stdout_line = process.stderr.readline()
        if stdout_line:
          try:
            sys.stderr.write(stdout_line.decode('utf-8'))
          except UnicodeDecodeError:
            pass
          stderr_log.write(stdout_line)
        stderr_line = process.stdout.readline()
        if stderr_line:
          try:
            sys.stdout.write(stderr_line.decode('utf-8'))
          except UnicodeDecodeError:
            pass
          stdout_log.write(stderr_line)

  process.communicate()
  returncode = process.returncode
  with open(workdir / f'returncode.txt', 'w') as f:
    f.write(str(returncode))
