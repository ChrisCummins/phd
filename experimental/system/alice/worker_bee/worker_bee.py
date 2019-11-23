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
"""This file contains TODO: one line summary."""
import pathlib
import time
import typing
from concurrent import futures

import grpc

from experimental.system.alice import alice_pb2
from experimental.system.alice import alice_pb2_grpc
from experimental.system.alice import bazel
from experimental.system.alice import git_repo
from labm8 import app
from labm8 import system

FLAGS = app.FLAGS

app.DEFINE_string('worker_bee_hostname', None, 'The hostname.')
app.DEFINE_integer('worker_bee_port', 5087, 'Port to listen for commands on.')
app.DEFINE_string('worker_bee_repo_root', str(
    pathlib.Path('~/phd').expanduser()), 'Path of worker bee root.')
app.DEFINE_string('worker_bee_output_root', '/var/phd/alice/outputs/',
                  'Path of worker bee root.')
app.DEFINE_string('ledger', 'localhost:5088', 'The path of the ledger service.')


class WorkerBee(alice_pb2_grpc.WorkerBeeServicer):

  def __init__(self, ledger: alice_pb2_grpc.LedgerStub, repo: git_repo.PhdRepo,
               output_dir: pathlib.Path):
    self._ledger = ledger
    self._repo = repo
    self._bazel = bazel.BazelClient(repo.path, output_dir)
    self._processes: typing.Dict[int, bazel.BazelRunProcess] = {}

    self.ledger.RegisterWorkerBee(
        alice_pb2.String(string=f'{self.hostname}:{FLAGS.worker_bee_port}'))

  def __del__(self):
    self.ledger.UnRegisterWorkerBee(
        alice_pb2.String(string=f'{self.hostname}:{FLAGS.worker_bee_port}'))

    for ledger_id, process in self._processes.items():
      if process.is_alive():
        app.Log(1, 'Waiting on job %d (process=%d) to finish', ledger_id,
                process.pid)
        process.join()
      else:
        self.EndProcess(ledger_id)

  def PruneCompletedProcesses(self) -> None:
    alive_processes = {}
    for ledger_id, process in self._processes:
      if process.is_alive():
        alive_processes[ledger_id] = process
      else:
        self.EndProcess(ledger_id)
    self._processes = alive_processes

  def EndProcess(self, ledger_id: int) -> None:
    process = self._processes[ledger_id]
    process.join(timeout=1)
    if process.is_alive():
      raise TypeError("EndProcess() called on process that is still alive.")
    self.ledger.Update(self.Get(alice_pb2.LedgerEntry(id=ledger_id), None))

  @property
  def hostname(self) -> str:
    return FLAGS.worker_bee_hostname or system.HOSTNAME

  @property
  def ledger(self):
    return self._ledger

  @property
  def repo(self) -> git_repo.PhdRepo:
    return self._repo

  @property
  def output_dir(self) -> pathlib.Path:
    return self._output_dir

  @property
  def bazel(self) -> bazel.BazelClient:
    return self._bazel

  def Run(self, request: alice_pb2.RunRequest, context) -> alice_pb2.Null:
    del context

    assert request.ledger_id not in self._processes

    # TODO: self.repo.FromRepoState(request.repo_state)
    process = self.bazel.Run(request)
    self._processes[request.ledger_id] = process

    return alice_pb2.Null()

  def Get(self, request: alice_pb2.LedgerId, context) -> alice_pb2.LedgerEntry:
    process = self._processes[request.id]
    returncode = process.returncode
    if returncode is None:
      outcome = alice_pb2.LedgerEntry.UNKNOWN
    elif returncode == 0:
      outcome = alice_pb2.LedgerEntry.RUN_SUCCEEDED
    else:
      outcome = alice_pb2.LedgerEntry.RUN_FAILED
    return alice_pb2.LedgerEntry(
        id=request.id,
        job_status=(alice_pb2.LedgerEntry.RUNNING
                    if process.is_alive() else alice_pb2.LedgerEntry.COMPLETE),
        job_outcome=outcome,
        stdout=process.stdout,
        stderr=process.stderr,
        returncode=returncode,
    )

  @classmethod
  def Main(cls, argv) -> None:
    """Return a main method for running this service.

    Returns:
      A callable main method.
    """
    if len(argv) > 1:
      raise app.UsageError('Unrecognized arguments')

    ledger_channel = grpc.insecure_channel(FLAGS.ledger)
    ledger = alice_pb2_grpc.LedgerStub(ledger_channel)

    repo = git_repo.PhdRepo(pathlib.Path(FLAGS.worker_bee_repo_root))
    outdir = pathlib.Path(FLAGS.worker_bee_output_root)
    outdir.mkdir(parents=True, exist_ok=True)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    service = cls(ledger, repo, outdir)
    alice_pb2_grpc.add_WorkerBeeServicer_to_server(service, server)

    port = FLAGS.worker_bee_port
    server.add_insecure_port(f'[::]:{port}')
    app.Log(1, '🐝  Listening for commands on %s. Buzz ...', port)
    server.start()
    try:
      while True:
        time.sleep(3600 * 24)
    except KeyboardInterrupt:
      server.stop(0)


if __name__ == '__main__':
  app.RunWithArgs(WorkerBee.Main)
