"""This file contains TODO: one line summary.

TODO: Detailed explanation of the file.
"""
import pathlib
import sys
import time

import grpc
from absl import app
from absl import flags

from alice import alice_pb2
from alice import alice_pb2_grpc
from alice import git_repo

FLAGS = flags.FLAGS

flags.DEFINE_string('phd', str(pathlib.Path("~/phd").expanduser()), 'Path.')
flags.DEFINE_string('ledger', 'localhost:5088', 'Ledger URL.')
flags.DEFINE_string('target', '', 'Target')
flags.DEFINE_list('bazel_args', [], 'Bazel args')
flags.DEFINE_list('bin_args', [], 'Binary args')
flags.DEFINE_integer('timeout_seconds', None, 'Timeout.')


def SummarizeJobStatus(ledger: alice_pb2.LedgerEntry):
  if ledger.HasField('job_outcome'):
    print(
        alice_pb2.LedgerEntry.JobStatus.Name(ledger.job_status),
        alice_pb2.LedgerEntry.JobOutcome.Name(ledger.job_outcome))


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

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
      ))

  print(f'Started job {job_id.id}')
  while True:
    status = stub.Get(job_id)
    SummarizeJobStatus(status)
    time.sleep(1)
    if status.job_status == alice_pb2.LedgerEntry.COMPLETE:
      sys.exit(status.returncode)


if __name__ == '__main__':
  app.run(main)
