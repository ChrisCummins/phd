"""Utility code for working with LLVM."""
import pathlib

from absl import app
from absl import flags

from compilers.llvm import opt
from experimental.compilers.reachability import control_flow_graph as cfg


FLAGS = flags.FLAGS


def ControlFlowGraphFromBytecode(
    bytecode_path: pathlib.Path) -> cfg.ControlFlowGraph:
  """Create a control flow graph from an LLVM bytecode file.

  Args:
    bytecode_path: Path of the bytecode file.

  Returns:
    A ControlFlowGraph instance.
  """
  # TODO(cec): Run opt foo.ll -dot-cfg, then parse cfg.main.dot and extract
  # control flow graph.
  # TODO(cec): Are multiple files created per bytecode file?
  # TODO(cec): Should we keep the basic block contents?
  raise NotImplementedError()
  opt.Exec(['-dot-cfg', str(bytecode_path)])


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))


if __name__ == '__main__':
  app.run(main)
