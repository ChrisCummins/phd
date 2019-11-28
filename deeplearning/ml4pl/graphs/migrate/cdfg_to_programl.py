"""Migrate networkx graphs to ProGraML protos.

See <github.com/ChrisCummins/ProGraML/issues/1>.
"""
import networkx as nx

from deeplearning.ml4pl.graphs.unlabelled import programl_pb2
from labm8.py import app

FLAGS = app.FLAGS


def CdfgGraphToProgramGraphProto(
  g: nx.MultiDiGraph,
) -> programl_pb2.ProgramGraph:
  # TODO(github.com/ChrisCummins/ProGraML/issues/1): Implement!
  del g


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))


if __name__ == "__main__":
  app.run(main)
