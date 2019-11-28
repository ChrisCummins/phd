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
"""Unit tests for //experimental/system/alice/ledger."""
import pathlib
from concurrent import futures

import grpc
import pytest

from experimental.system.alice import alice_pb2
from experimental.system.alice import alice_pb2_grpc
from experimental.system.alice.ledger import ledger
from labm8.py import test


@test.Fixture(scope="function")
def db(tempdir: pathlib.Path) -> ledger.Database:
  yield ledger.Database(f"sqlite:///{tempdir}/db")


@test.Fixture(scope="function")
def service(db: ledger.Database) -> ledger.LedgerService:
  yield ledger.LedgerService(db)


@test.Fixture(scope="function")
def stub(service: ledger.LedgerService) -> alice_pb2_grpc.LedgerStub:
  server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
  alice_pb2_grpc.add_LedgerServicer_to_server(service, server)

  port = server.add_insecure_port("[::]:0")
  server.start()

  channel = grpc.insecure_channel(f"localhost:{port}")
  stub = alice_pb2_grpc.LedgerStub(channel)
  yield stub
  server.stop(0)


@test.Fixture(scope="function")
def mock_run_request() -> alice_pb2.RunRequest:
  return alice_pb2.RunRequest(repo_state=None, target="//:foo",)


@test.Fixture(scope="function")
def mock_worker_bee() -> alice_pb2.String:
  """A test fixture which provides a connection to a mock worker bee."""

  class MockWorkerBee(alice_pb2_grpc.WorkerBeeServicer):
    """This mock service does nothing."""

    def Run(self, request: alice_pb2.RunRequest, context) -> alice_pb2.Null:
      """Mock Run() which does nothing."""
      del request
      del context
      return alice_pb2.Null()

    def Get(self, request: alice_pb2.LedgerId, context) -> alice_pb2.Null:
      """Mock Run() which does nothing."""
      del request
      del context
      return alice_pb2.LedgerEntry()

  server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
  alice_pb2_grpc.add_WorkerBeeServicer_to_server(MockWorkerBee(), server)

  port = server.add_insecure_port("[::]:0")
  server.start()

  yield alice_pb2.String(string=f"localhost:{port}")
  server.stop(0)


def test_LedgerService_Add_no_worker_bees(
  stub: alice_pb2_grpc.LedgerStub, mock_run_request: alice_pb2.RunRequest
):
  """Test that Ledger add fails when no worker bees are registered."""
  with test.Raises(Exception) as e_ctx:
    stub.Add(mock_run_request)

  assert "No worker bees available" in str(e_ctx.value)


def test_LedgerService_RegisterWorkerBee_smoke_test(
  stub: alice_pb2_grpc.LedgerStub, mock_worker_bee: alice_pb2.String
):
  """Worker bee can be registered."""
  stub.RegisterWorkerBee(mock_worker_bee)
  stub.RegisterWorkerBee(mock_worker_bee)


def test_LedgerService_Add_sequential_ids(
  stub: alice_pb2_grpc.LedgerStub,
  mock_worker_bee: alice_pb2.String,
  mock_run_request: alice_pb2.RunRequest,
):
  """Ledger entry IDs are sequential."""
  stub.RegisterWorkerBee(mock_worker_bee)

  ledger_id = stub.Add(mock_run_request)
  assert ledger_id.id == 1

  ledger_id = stub.Add(mock_run_request)
  assert ledger_id.id == 2

  ledger_id = stub.Add(mock_run_request)
  assert ledger_id.id == 3


def test_LedgerSerivce_Get_id(
  stub: alice_pb2_grpc.LedgerStub,
  mock_worker_bee: alice_pb2.String,
  mock_run_request: alice_pb2.RunRequest,
):
  stub.RegisterWorkerBee(mock_worker_bee)
  ledger_id = stub.Add(mock_run_request)

  entry = stub.Get(ledger_id)
  assert ledger_id.id == entry.id


def test_LedgerSerivce_Update_stdout(
  stub: alice_pb2_grpc.LedgerStub,
  mock_worker_bee: alice_pb2.String,
  mock_run_request: alice_pb2.RunRequest,
):
  """Check that stdout can be set and got."""
  stub.RegisterWorkerBee(mock_worker_bee)
  ledger_id = stub.Add(mock_run_request)

  stub.Update(alice_pb2.LedgerEntry(id=ledger_id.id, stdout="Hello, stdout!"))
  entry = stub.Get(ledger_id)
  assert entry.stdout == "Hello, stdout!"

  stub.Update(alice_pb2.LedgerEntry(id=ledger_id.id, stdout="abcd"))
  entry = stub.Get(ledger_id)
  assert entry.stdout == "abcd"


def test_LedgerSerivce_Update_stderr(
  stub: alice_pb2_grpc.LedgerStub,
  mock_worker_bee: alice_pb2.String,
  mock_run_request: alice_pb2.RunRequest,
):
  """Check that stderr can be set and got."""
  stub.RegisterWorkerBee(mock_worker_bee)
  ledger_id = stub.Add(mock_run_request)

  stub.Update(alice_pb2.LedgerEntry(id=ledger_id.id, stderr="Hello, stderr!"))
  entry = stub.Get(ledger_id)
  assert entry.stderr == "Hello, stderr!"

  stub.Update(alice_pb2.LedgerEntry(id=ledger_id.id, stderr="abcd"))
  entry = stub.Get(ledger_id)
  assert entry.stderr == "abcd"


def test_LedgerSerivce_Update_returncode(
  stub: alice_pb2_grpc.LedgerStub,
  mock_worker_bee: alice_pb2.String,
  mock_run_request: alice_pb2.RunRequest,
):
  """Check that returncode can be set and got."""
  stub.RegisterWorkerBee(mock_worker_bee)
  ledger_id = stub.Add(mock_run_request)

  stub.Update(alice_pb2.LedgerEntry(id=ledger_id.id, returncode=123))
  entry = stub.Get(ledger_id)
  assert entry.returncode == 123


if __name__ == "__main__":
  test.Main()
