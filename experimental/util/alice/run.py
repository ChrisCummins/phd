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
"""Run a command."""
import pathlib
import sys
import time

import grpc

from experimental.util.alice import alice_pb2
from experimental.util.alice import alice_pb2_grpc
from experimental.util.alice import git_repo
from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_string("phd", str(pathlib.Path("~/phd").expanduser()), "Path.")
app.DEFINE_string("ledger", "localhost:5088", "Ledger URL.")
app.DEFINE_string("target", "", "Target")
app.DEFINE_list("bazel_args", [], "Bazel args")
app.DEFINE_list("bin_args", [], "Binary args")
app.DEFINE_integer("timeout_seconds", None, "Timeout.")


def SummarizeJobStatus(ledger: alice_pb2.LedgerEntry):
  if ledger.HasField("job_outcome"):
    print(
      alice_pb2.LedgerEntry.JobStatus.Name(ledger.job_status),
      alice_pb2.LedgerEntry.JobOutcome.Name(ledger.job_outcome),
    )


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))

  channel = grpc.insecure_channel(FLAGS.ledger)
  stub = alice_pb2_grpc.LedgerStub(channel)

  phd_path = pathlib.Path(FLAGS.phd)
  repo = git_repo.PhdRepo(phd_path)

  # TODO(cec): Sanity check values.
  target = FLAGS.target
  bazel_args = FLAGS.bazel_args
  bin_args = FLAGS.bin_args
  timeout_seconds = FLAGS.timeout_seconds

  job_id = stub.Add(
    alice_pb2.RunRequest(
      repo_state=repo.ToRepoState(),
      target=target,
      bazel_args=bazel_args,
      bin_args=bin_args,
      timeout_seconds=timeout_seconds,
    )
  )

  print(f"Started job {job_id.id}")
  while True:
    status = stub.Get(job_id)
    SummarizeJobStatus(status)
    time.sleep(1)
    if status.job_status == alice_pb2.LedgerEntry.COMPLETE:
      sys.exit(status.returncode)


if __name__ == "__main__":
  app.RunWithArgs(main)
