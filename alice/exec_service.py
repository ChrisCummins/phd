"""The exec service runs jobs on a local machine."""
import pathlib

from absl import app
from absl import flags

from alice import alice_pb2_grpc


FLAGS = flags.FLAGS


class ExecService(alice_pb2_grpc.ExecServiceServicer):

  def __init__(self, repo_root: pathlib.Path):
    pass

  def Run(self, request: alice_pb2_grpc.RunRequest,
          context) -> alice_pb2_grpc.RunResponse:
    raise NotImplementedError


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))


if __name__ == '__main__':
  app.run(main)
