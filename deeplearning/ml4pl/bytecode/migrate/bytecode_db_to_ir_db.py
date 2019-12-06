"""Migrate old bytecode table to new IR schema.

See <github.com/ChrisCummins/ProGraML/issues/6>.
"""
import multiprocessing
from typing import Iterable
from typing import List

import sqlalchemy as sql

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.ir import ir_database
from labm8.py import app
from labm8.py import humanize
from labm8.py import ppar
from labm8.py import progress
from labm8.py import sqlutil


FLAGS = app.FLAGS

app.DEFINE_database(
  "bytecode_db",
  bytecode_database.Database,
  None,
  "The database of bytecodes to migrate.",
  must_exist=True,
)
app.DEFINE_integer(
  "nproc", multiprocessing.cpu_count(), "The number of processes to spawn."
)
app.DEFINE_integer(
  "max_reader_queue_size", 3, "Tuning parameter. The maximum number of proto."
)
app.DEFINE_integer(
  "batch_size", 512, "Tuning parameter. The number of bytecodes to read."
)
app.DEFINE_integer(
  "chunk_size", 32, "Tuning parameter. The chunksize for batches."
)


def BytecodeToIntermediateRepresentation(
  bytecode: bytecode_database.LlvmBytecode,
) -> ir_database.IntermediateRepresentation:
  """Convert a bytecode to an intermediate representation."""
  # Map string language to enum.
  source_language = {
    "c": ir_database.SourceLanguage.C,
    "cpp": ir_database.SourceLanguage.CPP,
    "opencl": ir_database.SourceLanguage.OPENCL,
    "swift": ir_database.SourceLanguage.SWIFT,
    "haskell": ir_database.SourceLanguage.HASKELL,
    "fortran": ir_database.SourceLanguage.FORTRAN,
  }[bytecode.language.lower()]
  # Haskell uses an older version of LLVM.
  type = (
    ir_database.IrType.LLVM_3_5
    if bytecode.language.lower() == "haskell"
    else ir_database.IrType.LLVM_6_0
  )

  if bytecode.clang_returncode == 0:
    ir = ir_database.IntermediateRepresentation.CreateFromText(
      source=bytecode.source_name,
      relpath=bytecode.relpath,
      source_language=source_language,
      type=type,
      cflags=bytecode.cflags,
      text=bytecode.bytecode,
    )
  else:
    ir = ir_database.IntermediateRepresentation.CreateEmpty(
      source=bytecode.source_name,
      relpath=bytecode.relpath,
      source_language=source_language,
      type=type,
      cflags=bytecode.cflags,
    )
  ir.id = bytecode.id

  return ir


def BatchedBytecodeReader(
  bytecode_db: bytecode_database.Database,
  ids_to_read: List[int],
  batch_size: int,
  ctx: progress.ProgressContext,
) -> Iterable[List[bytecode_database.LlvmBytecode]]:
  """Read from the given list of graph IDs in batches."""
  ids_to_read = list(sorted(ids_to_read))
  with bytecode_db.Session() as session:
    for i in range(0, len(ids_to_read), batch_size):
      ids_batch = ids_to_read[i : i + batch_size]
      with ctx.Profile(1, f"[reader] Read {len(ids_batch)} bytecodes"):
        bytecodes = (
          session.query(bytecode_database.LlvmBytecode)
          .filter(bytecode_database.LlvmBytecode.id >= ids_batch[0])
          .filter(bytecode_database.LlvmBytecode.id <= ids_batch[-1])
          .all()
        )
      yield bytecodes


class Migrator(progress.Progress):
  """Thread to migrate graph databases"""

  def __init__(
    self, bytecode_db: bytecode_database.Database, ir_db: ir_database.Database,
  ):
    self.bytecode_db = bytecode_db
    self.ir_db = ir_db

    # Setup: Get the IDs of the graphs to process.

    # Get graphs that have already been processed.
    with self.ir_db.Session() as out_session:
      already_done_max, already_done_count = out_session.query(
        sql.func.max(ir_database.IntermediateRepresentation.id),
        sql.func.count(ir_database.IntermediateRepresentation.id),
      ).one()
      already_done_max = already_done_max or -1

    # Get the total number of bytecodes to process.
    with self.bytecode_db.Session() as in_session:
      total_graph_count = in_session.query(
        sql.func.count(bytecode_database.LlvmBytecode.id)
      ).scalar()
      ids_to_do = [
        row.id
        for row in in_session.query(bytecode_database.LlvmBytecode.id)
        .filter(bytecode_database.LlvmBytecode.id > already_done_max)
        .order_by(bytecode_database.LlvmBytecode.id)
      ]

    # Sanity check.
    if len(ids_to_do) + already_done_count != total_graph_count:
      app.FatalWithoutStackTrace(
        "ids_to_do(%s) + already_done(%s) != total_rows(%s)",
        len(ids_to_do),
        already_done_count,
        total_graph_count,
      )

    with self.ir_db.Session(commit=True) as out_session:
      out_session.add(
        ir_database.Meta.Create(
          key="Bytecode migration counts",
          value=(already_done_count, total_graph_count),
        )
      )
    app.Log(
      1, "Selected %s to process", humanize.Plural(len(ids_to_do), "bytecode")
    )

    super(Migrator, self).__init__(
      name="migrate",
      i=already_done_count,
      n=total_graph_count,
      unit="bytecodes",
    )
    self.bytecode_reader = ppar.ThreadedIterator(
      BatchedBytecodeReader(
        self.bytecode_db,
        ids_to_do,
        FLAGS.batch_size,
        ctx=self.ctx.ToProgressContext(),
      ),
      max_queue_size=FLAGS.max_reader_queue_size,
    )

  def Run(self):
    """Run the migration"""
    pool = multiprocessing.Pool(FLAGS.nproc)

    with sqlutil.BufferedDatabaseWriter(
      self.ir_db,
      max_buffer_size=128 * 1024 * 1024,
      max_buffer_length=4096,
      log_level=1,
      ctx=self.ctx.ToProgressContext(),
    ) as writer:
      # Main loop: Peel off chunks of bytecodes to process, process them, and
      # write the results to the output database.
      for bytecode_batch in self.bytecode_reader:
        self.ctx.i += len(bytecode_batch)

        for ir in pool.imap_unordered(
          BytecodeToIntermediateRepresentation,
          bytecode_batch,
          chunksize=FLAGS.chunk_size,
        ):
          writer.AddOne(ir)


def main():
  """Main entry point."""
  bytecode_db = FLAGS.bytecode_db()
  ir_db = FLAGS.ir_db()

  progress.Run(Migrator(bytecode_db, ir_db))


if __name__ == "__main__":
  app.Run(main)
