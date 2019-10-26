"""Populate the table of control flow graphs from the bytecodes table."""
import multiprocessing
import time
import typing

import progressbar
import pyparsing
from labm8 import app

from compilers.llvm import opt
from compilers.llvm import opt_util
from deeplearning.ml4pl import ml4pl_pb2
from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs.unlabelled.cfg import control_flow_graph as cfg
from deeplearning.ml4pl.graphs.unlabelled.cfg import llvm_util

FLAGS = app.FLAGS

app.DEFINE_database('bytecode_db', bytecode_database.Database, None,
                    'Path of database to populate.')
app.DEFINE_integer(
    'nproc', multiprocessing.cpu_count(),
    'The number of parallel processes to use when creating '
    'the dataset.')


def CreateControlFlowGraphsFromLlvmBytecode(
    bytecode_tuple: typing.Tuple[int, str]
) -> typing.List[ml4pl_pb2.ControlFlowGraphFromLlvmBytecode]:
  """Parallel worker process for extracting CFGs from bytecode."""
  # Expand the input tuple.
  bytecode_id, bytecode = bytecode_tuple
  protos = []

  # Extract the dot sources from the bytecode.
  dot_generator = opt_util.DotControlFlowGraphsFromBytecode(bytecode)
  cfg_id = 0
  # We use a while loop here rather than iterating over the dot_generator
  # directly because the dot_generator can raise an exception, and we don't
  # want to lose all of the dot files if only one of them would raise an
  # exception.
  while True:
    try:
      dot = next(dot_generator)
      # Instantiate a CFG from the dot source.
      graph = llvm_util.ControlFlowGraphFromDotSource(dot)
      graph.ValidateControlFlowGraph(strict=False)
      protos.append(
          ml4pl_pb2.ControlFlowGraphFromLlvmBytecode(
              bytecode_id=bytecode_id,
              cfg_id=cfg_id,
              control_flow_graph=graph.ToProto(),
              status=0,
              error_message='',
              block_count=graph.number_of_nodes(),
              edge_count=graph.number_of_edges(),
              is_strict_valid=graph.IsValidControlFlowGraph(strict=True)))
    except (UnicodeDecodeError, cfg.MalformedControlFlowGraphError, ValueError,
            opt.OptException, pyparsing.ParseException) as e:
      protos.append(
          ml4pl_pb2.ControlFlowGraphFromLlvmBytecode(
              bytecode_id=bytecode_id,
              cfg_id=cfg_id,
              control_flow_graph=ml4pl_pb2.ControlFlowGraph(),
              status=1,
              error_message=str(e),
              block_count=0,
              edge_count=0,
              is_strict_valid=False,
          ))
    except StopIteration:
      # Stop once the dot generator has nothing more to give.
      break

    cfg_id += 1

  return protos


def PopulateControlFlowGraphTable(db: bytecode_database.Database,
                                  n: int = 10000) -> bool:
  """Populate the control flow graph table from LLVM bytecodes.

  For each row in the LlvmBytecode table, we ectract the CFGs and add them
  to the ControlFlowGraphProto table.
  """
  with db.Session(commit=True) as s:
    # We only process bytecodes which are not already in the CFG table.
    already_done_ids = s.query(
        bytecode_database.ControlFlowGraphProto.bytecode_id)
    # Query that returns (id,bytecode) tuples for all bytecode files that were
    # successfully generated and have not been entered into the CFG table yet.
    todo = s.query(bytecode_database.LlvmBytecode.id, bytecode_database.LlvmBytecode.bytecode) \
      .filter(bytecode_database.LlvmBytecode.clang_returncode == 0) \
      .filter(~bytecode_database.LlvmBytecode.id.in_(already_done_ids)) \
      .limit(n)

    count = todo.count()
    if not count:
      return False

    bar = progressbar.ProgressBar()
    bar.max_value = count

    # Process bytecodes in parallel.
    pool = multiprocessing.Pool(FLAGS.nproc)
    last_commit_time = time.time()
    for i, protos in enumerate(
        pool.imap_unordered(CreateControlFlowGraphsFromLlvmBytecode, todo)):
      the_time = time.time()
      bar.update(i)

      # Each bytecode file can produce multiple CFGs. Construct table rows from
      # the protos returned.
      rows = [
          bytecode_database.ControlFlowGraphProto(
              **bytecode_database.ControlFlowGraphProto.FromProto(proto))
          for proto in protos
      ]

      # Add the rows in a batch.
      s.add_all(rows)

      # Commit every 10 seconds.
      if the_time - last_commit_time > 10:
        s.commit()
        last_commit_time = the_time

  return True


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  db = FLAGS.bytecode_db()
  while PopulateControlFlowGraphTable(db):
    print()


if __name__ == '__main__':
  app.RunWithArgs(main)
