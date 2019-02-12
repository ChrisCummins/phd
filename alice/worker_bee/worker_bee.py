"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
import pathlib
import shutil
import time
import typing
from concurrent import futures

import grpc
from absl import app
from absl import flags
from absl import logging

from alice import alice_pb2
from alice import alice_pb2_grpc
from alice import bazel
from alice import git_repo
from labm8 import system


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'worker_bee_hostname', None,
    'The hostname.')
flags.DEFINE_integer(
    'worker_bee_port', 5087,
    'Port to listen for commands on.')
flags.DEFINE_string(
    'worker_bee_repo_root', str(pathlib.Path('~/phd').expanduser()),
    'Path of worker bee root.')
flags.DEFINE_string(
    'worker_bee_output_root', '/var/phd/alice/outputs/',
    'Path of worker bee root.')
flags.DEFINE_string(
    'ledger', 'localhost:5088',
    'The path of the ledger service.')


class WorkerBee(alice_pb2_grpc.WorkerBeeServicer):

  def __init__(self, ledger: alice_pb2_grpc.LedgerStub,
               repo: git_repo.PhdRepo, output_dir: pathlib.Path):
    self._ledger = ledger
    self._repo = repo
    self._bazel = bazel.BazelClient(repo.path)
    self._processes: typing.List[bazel.BazelRunProcess] = []
    self._output_dir = output_dir

    self.ledger.RegisterWorkerBee(
        alice_pb2.String(string=f'{self.hostname}:{FLAGS.worker_bee_port}'))

  def __del__(self):
    self.ledger.UnRegisterWorkerBee(
        alice_pb2.String(string=f'{self.hostname}:{FLAGS.worker_bee_port}'))

    for process in self._processes:
      if process.is_alive():
        logging.info('Waiting on job before finishing %d', process.id)
        process.join()
      else:
        self.EndProcess(process)

  def PruneCompletedProcesses(self) -> None:
    alive_processes = []
    for process in self._processes:
      if process.is_alive():
        alive_processes.append(process)
      else:
        self.EndProcess(process)
    self._processes = alive_processes

  def EndProcess(self, process: bazel.BazelRunProcess) -> None:
    assert (process.workdir / 'stdout.txt').is_file()
    assert (process.workdir / 'stderr.txt').is_file()
    assert (process.workdir / 'returncode.txt').is_file()
    shutil.rmtree(process.workdir)

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

  def Run(self, request: alice_pb2.RunRequest,
          context) -> alice_pb2.Null:
    del context

    # TODO: self.repo.FromRepoState(request.repo_state)
    workdir = self.output_dir / str(request.ledger_id)
    assert not workdir.is_dir()
    workdir.mkdir()
    process = self.bazel.Run(request, workdir)
    self._processes.append(process)

    return alice_pb2.Null()

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
    logging.info('🐝  Listening for commands on %s. Buzz ...', port)
    server.start()
    try:
      while True:
        time.sleep(3600 * 24)
    except KeyboardInterrupt:
      server.stop(0)


if __name__ == '__main__':
  app.run(WorkerBee.Main)
