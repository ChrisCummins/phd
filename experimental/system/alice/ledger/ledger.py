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
"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
import datetime
import random
import time
import typing
from concurrent import futures

import config_pb2
import grpc
import sqlalchemy as sql
from sqlalchemy.ext import declarative

from experimental.system.alice import alice_pb2
from experimental.system.alice import alice_pb2_grpc
from labm8.py import app
from labm8.py import labdate
from labm8.py import sqlutil

FLAGS = app.FLAGS

app.DEFINE_string("ledger_db", "sqlite:////tmp/ledger.db", "Ledger database.")
app.DEFINE_integer("ledger_port", 5088, "Port to service.")
app.DEFINE_integer("ledger_service_thread_count", 10, "Number of threads.")

Base = declarative.declarative_base()


class LedgerEntry(
  Base, sqlutil.TablenameFromCamelCapsClassNameMixin, sqlutil.ProtoBackedMixin
):
  """Ledger store."""

  proto_t = alice_pb2.LedgerEntry

  # A numeric counter of this entry in the ledger. >= 1.
  id: int = sql.Column(sql.Integer, primary_key=True, autoincrement=True)

  worker_id: str = sql.Column(sql.String(512))

  # GlobalConfig fields.
  uname: str = sql.Column(sql.String(32))
  configure_id: str = sql.Column(sql.String(64))
  with_cuda: bool = sql.Column(sql.Boolean)
  repo_root: str = sql.Column(sql.String(255))

  repo_remote_url: str = sql.Column(sql.String(255), nullable=False)
  repo_tracking_branch: str = sql.Column(sql.String(128), nullable=False)
  repo_head_id: str = sql.Column(sql.String(64), nullable=False)
  # TODO: repo_git_diff: str = ref to big table.

  target: str = sql.Column(sql.String(255), nullable=False)
  bazel_args: str = sql.Column(sql.String(4096), nullable=False)
  bin_args: str = sql.Column(sql.String(4096), nullable=False)
  timeout_seconds: str = sql.Column(sql.String(255))

  job_status: int = sql.Column(sql.Integer, nullable=False)
  job_outcome: int = sql.Column(sql.Integer)

  build_started: datetime.datetime = sql.Column(
    sql.DateTime, nullable=False, default=datetime.datetime.utcnow
  )
  run_started: datetime.datetime = sql.Column(sql.DateTime)
  run_end: datetime.datetime = sql.Column(sql.DateTime)

  returncode: int = sql.Column(sql.Integer)

  # TODO(cec): Add references to stderr and assets tables.

  @classmethod
  def FromProto(
    cls, proto: alice_pb2.LedgerEntry
  ) -> typing.Dict[str, typing.Any]:
    return {
      "id": proto.id if proto.id else None,
      "worker_id": proto.worker_id if proto.worker_id else None,
      "uname": proto.repo_config.uname,
      "configure_id": proto.repo_config.configure_id,
      "with_cuda": proto.repo_config.with_cuda,
      "repo_root": proto.repo_config.paths.repo_root,
      "repo_remote_url": proto.run_request.repo_state.remote_url,
      "repo_tracking_branch": proto.run_request.repo_state.tracking_branch,
      "repo_head_id": proto.run_request.repo_state.head_id,
      "target": proto.run_request.target,
      "bazel_args": " ".join(proto.run_request.bazel_args),
      "bin_args": " ".join(proto.run_request.bin_args),
      "timeout_seconds": proto.run_request.timeout_seconds,
      "job_status": proto.job_status,
      "job_outcome": proto.job_outcome,
      "returncode": proto.returncode,
    }

  def ToProto(self) -> alice_pb2.LedgerEntry:
    return alice_pb2.LedgerEntry(
      id=self.id,
      worker_id=self.worker_id,
      repo_config=config_pb2.GlobalConfig(
        uname=self.uname,
        configure_id=self.configure_id,
        with_cuda=self.with_cuda,
        options=config_pb2.GlobalConfigOptions(
          with_cuda=False,
          update_git_submodules=False,
          symlink_python=False,
          install_git_hooks=False,
        ),
        paths=config_pb2.GlobalConfigPaths(
          repo_root=self.repo_root, python="",
        ),
      ),
      # TODO: Run request?
      job_status=self.job_status,
      job_outcome=self.job_outcome,
      build_start_unix_epoch_ms=labdate.MillisecondsTimestamp(
        self.build_started
      ),
      run_start_unix_epoch_ms=labdate.MillisecondsTimestamp(self.run_started),
      run_end_unix_epoch_ms=labdate.MillisecondsTimestamp(self.run_end),
      returncode=self.returncode,
    )


class StdoutString(Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
  """Ledger outputs."""

  ledger_id: int = sql.Column(
    sql.Integer,
    sql.ForeignKey(LedgerEntry.id),
    nullable=False,
    primary_key=True,
  )
  string: str = sql.Column(
    sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=False
  )


class StderrString(Base, sqlutil.TablenameFromCamelCapsClassNameMixin):
  """Ledger outputs."""

  ledger_id: int = sql.Column(
    sql.Integer,
    sql.ForeignKey(LedgerEntry.id),
    nullable=False,
    primary_key=True,
  )
  string: str = sql.Column(
    sqlutil.ColumnTypes.UnboundedUnicodeText(), nullable=False
  )


class Database(sqlutil.Database):
  def __init__(self, url: str):
    super(Database, self).__init__(url, Base)


class LedgerService(alice_pb2_grpc.LedgerServicer):
  def __init__(self, db: Database):
    self._db = db
    self.worker_bees: typing.Dict[str, alice_pb2_grpc.WorkerBeeStub] = {}

  @property
  def db(self) -> Database:
    return self._db

  def SelectWorkerIdForRunRequest(
    self, run_request: alice_pb2.RunRequest
  ) -> alice_pb2_grpc.WorkerBeeStub:
    if not self.worker_bees:
      raise ValueError("No worker bees available")

    if run_request.force_worker_id:
      return self.worker_bees[run_request.force_worker_id]
    else:
      return random.choice(list(self.worker_bees.values()))

  def RegisterWorkerBee(
    self, request: alice_pb2.String, context
  ) -> alice_pb2.Null:
    del context

    app.Log(1, "Worker bee %s registered", request.string)
    channel = grpc.insecure_channel(request.string)
    worker_bee = alice_pb2_grpc.WorkerBeeStub(channel)
    worker_bee.id = request.string
    if request.string in self.worker_bees:
      del self.worker_bees[request.string]
    self.worker_bees[request.string] = worker_bee
    return alice_pb2.Null()

  def UnRegisterWorkerBee(
    self, request: alice_pb2.String, context
  ) -> alice_pb2.Null:
    del context

    app.Log(1, "Worker bee %s unregistered", request.string)
    if request.string in self.worker_bees:
      del self.worker_bees[request.string]
    return alice_pb2.Null()

  def Add(self, request: alice_pb2.RunRequest, context) -> alice_pb2.LedgerId:
    del context

    with self.db.Session(commit=True) as s:
      entry = LedgerEntry(
        **LedgerEntry.FromProto(
          alice_pb2.LedgerEntry(
            job_status=alice_pb2.LedgerEntry.BUILDING, run_request=request,
          )
        )
      )

      s.add(entry)
      s.flush()

      entry_id = entry.id
      app.Log(1, "Created new ledger entry %s", entry_id)

      worker_bee = self.SelectWorkerIdForRunRequest(request)
      entry.worker_id = worker_bee.id
      request.ledger_id = entry_id
      worker_bee.Run(request, None)

    return alice_pb2.LedgerId(id=entry_id)

  def Update(self, request: alice_pb2.LedgerEntry, context) -> alice_pb2.Null:
    with self.db.Session(commit=True) as session:
      ledger_entry = (
        session.query(LedgerEntry).filter(LedgerEntry.id == request.id).one()
      )

      update_dict = LedgerEntry.FromProto(request)
      for key, value in update_dict.items():
        if value:
          setattr(ledger_entry, key, value)

      # Update stdout and stderr.
      if request.HasField("stdout"):
        stdout = session.GetOrAdd(StdoutString, ledger_id=ledger_entry.id)
        stdout.string = request.stdout

      if request.HasField("stderr"):
        stderr = session.GetOrAdd(StderrString, ledger_id=ledger_entry.id)
        stderr.string = request.stderr

    return alice_pb2.Null()

  def Get(self, request: alice_pb2.LedgerId, context) -> alice_pb2.LedgerId:
    del context

    with self.db.Session(commit=False) as session:
      ledger_entry = (
        session.query(LedgerEntry).filter(LedgerEntry.id == request.id).one()
      )

      if ledger_entry != alice_pb2.LedgerEntry.COMPLETE:
        # Ping for an update.
        update = self.worker_bees[ledger_entry.worker_id].Get(
          alice_pb2.LedgerId(id=ledger_entry.id)
        )
        update_dict = LedgerEntry.FromProto(update)
        for key, value in update_dict.items():
          if value:
            setattr(ledger_entry, key, value)

      proto = ledger_entry.ToProto()

      # Add joined stdout and stderr.
      stdout = (
        session.query(StdoutString)
        .filter(StdoutString.ledger_id == request.id)
        .first()
      )
      if stdout:
        proto.stdout = stdout.string

      stderr = (
        session.query(StderrString)
        .filter(StderrString.ledger_id == request.id)
        .first()
      )
      if stderr:
        proto.stderr = stderr.string

    return proto

  @classmethod
  def Main(cls, argv) -> None:
    """Return a main method for running this service.

    Returns:
      A callable main method.
    """
    if len(argv) > 1:
      raise app.UsageError("Unrecognized arguments")

    thread_pool = futures.ThreadPoolExecutor(
      max_workers=FLAGS.ledger_service_thread_count
    )
    server = grpc.server(thread_pool, options=(("grpc.so_reuseport", 0),))
    service = cls(Database(FLAGS.ledger_db))
    alice_pb2_grpc.add_LedgerServicer_to_server(service, server)

    port = FLAGS.ledger_port
    port = server.add_insecure_port(f"[::]:{port}")
    app.Log(1, "📜  Listening for commands on %s ...", port)
    server.start()
    try:
      while True:
        time.sleep(3600 * 24)
    except KeyboardInterrupt:
      server.stop(0)


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))


if __name__ == "__main__":
  app.RunWithArgs(LedgerService.Main)
