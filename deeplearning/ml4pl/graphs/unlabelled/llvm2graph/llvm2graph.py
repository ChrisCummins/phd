"""Construct a ProGraML graph from LLVM intermediate representation."""
from deeplearning.ml4pl.graphs.unlabelled.llvm2graph import graph_builder
from labm8.py import app

FLAGS = app.FLAGS


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))

  # TODO(github.com/ChrisCummins/ProGraML/issues/2): Implement!
  bytecode = ""
  opt = ""

  builder = graph_builder.ProGraMLGraphBuilder()
  graph_proto = builder.Build(bytecode, opt)
  print(graph_proto)


if __name__ == "__main__":
  app.Run(main)
