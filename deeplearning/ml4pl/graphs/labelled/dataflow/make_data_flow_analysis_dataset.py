"""This module prepares datasets for data flow analyses."""
import multiprocessing
import pathlib
import signal
import sys
import time
import traceback
from typing import Iterable
from typing import List
from typing import NamedTuple
from typing import Tuple

import networkx as nx
import sqlalchemy as sql

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from deeplearning.ml4pl.graphs.labelled.dataflow.alias_set import alias_set
from deeplearning.ml4pl.graphs.labelled.dataflow.datadep import data_dependence
from deeplearning.ml4pl.graphs.labelled.dataflow.domtree import dominator_tree
from deeplearning.ml4pl.graphs.labelled.dataflow.liveness import liveness
from deeplearning.ml4pl.graphs.labelled.dataflow.polyhedra import polyhedra
from deeplearning.ml4pl.graphs.labelled.dataflow.reachability import (
  reachability,
)
from deeplearning.ml4pl.graphs.labelled.dataflow.subexpressions import (
  subexpressions,
)
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from labm8.py import app
from labm8.py import humanize
from labm8.py import ppar
from labm8.py import progress
from labm8.py import sqlutil

app.DEFINE_string("analysis", None, "The name of the analysis to run.")
# TODO(github.com/ChrisCummins/ProGraML/issues/22): Port all graph annotators
# to new graph tuple representation.
# app.DEFINE_database(
#   "bytecode_db",
#   bytecode_database.Database,
#   None,
#   "URL of database to read bytecode from. Only required when "
#   "analysis requires bytecode.",
#   must_exist=True,
# )
app.DEFINE_integer(
  "max_instances_per_graph",
  10,
  "The maximum number of instances to produce from a single input graph. "
  "For a graph with `n` root statements, `n` instances can be produced by "
  "changing the root statement.",
)
app.DEFINE_integer(
  "annotator_timeout",
  120,
  "The maximum number of seconds to allow an annotator to process a single "
  "graph.",
)
app.DEFINE_integer(
  "max_instances", 0, "If set, limit the number of processed instances."
)

app.DEFINE_integer(
  "nproc",
  multiprocessing.cpu_count(),
  "Tuning parameter. The number of processes to spawn.",
)
app.DEFINE_integer(
  "proto_batch_mb",
  4,
  "Tuning parameter. The number of megabytes of protocol buffers to read in "
  "a batch.",
)
app.DEFINE_integer(
  "max_reader_queue_size",
  3,
  "Tuning parameter. The maximum number of proto chunks to read ahead of the "
  "workers.",
)
app.DEFINE_integer(
  "max_tasks_per_worker",
  64,
  "Tuning parameter. The maximum number of tasks for a worker to process "
  "before restarting.",
)
app.DEFINE_integer(
  "chunk_size",
  32,
  "Tuning parameter. The number of protos to assign to each worker.",
)
app.DEFINE_integer(
  "write_buffer_mb",
  32,
  "Tuning parameter. The size of the write buffer, in megabytes.",
)
app.DEFINE_integer(
  "write_buffer_length",
  4096,
  "Tuning parameter. The maximum length of the write buffer.",
)

FLAGS = app.FLAGS


def GetDataFlowGraphAnnotator(
  name: str,
) -> data_flow_graphs.DataFlowGraphAnnotator:
  """Get the data flow annotator."""
  # TODO(github.com/ChrisCummins/ProGraML/issues/22): Port all graph annotators
  # to new graph tuple representation.
  if name == "reachability":
    return reachability.ReachabilityAnnotator()
  else:
    raise app.UsageError(f"Unknown analyses '{name}'")


class ProgramGraphProto(NamedTuple):
  """A serialized program graph protocol buffer."""

  ir_id: int
  serialized_proto: bytes


def BatchedProtoReader(
  proto_db: unlabelled_graph_database.Database,
  ids_and_sizes_to_do: List[Tuple[int, int]],
  batch_size_in_bytes: int,
  ctx: progress.ProgressBarContext,
) -> Iterable[List[ProgramGraphProto]]:
  """Read from the given list of IDs in batches."""
  ids_and_sizes_to_do = sorted(ids_and_sizes_to_do, key=lambda x: x[0])
  i = 0
  while i < len(ids_and_sizes_to_do):
    end_i = i
    batch_size = 0
    while batch_size < batch_size_in_bytes:
      batch_size += ids_and_sizes_to_do[end_i][1]
      end_i += 1
      if end_i >= len(ids_and_sizes_to_do):
        # We have run out of graphs to read.
        break

    with proto_db.Session() as session:
      with ctx.Profile(
        2,
        f"[reader] Read {humanize.BinaryPrefix(batch_size, 'B')} "
        f"batch of {end_i - i} unlabelled graphs",
      ):
        start_id = ids_and_sizes_to_do[i][0]
        end_id = ids_and_sizes_to_do[end_i - 1][0]
        graphs = (
          session.query(unlabelled_graph_database.ProgramGraph)
          .filter(unlabelled_graph_database.ProgramGraph.ir_id >= start_id)
          .filter(unlabelled_graph_database.ProgramGraph.ir_id <= end_id)
          .options(
            sql.orm.joinedload(unlabelled_graph_database.ProgramGraph.data)
          )
          .all()
        )
      yield [
        ProgramGraphProto(
          ir_id=graph.ir_id, serialized_proto=graph.data.serialized_proto
        )
        for graph in graphs
      ]

    i = end_i


def ProcessOneProto(
  analysis: str,
  program_graph: ProgramGraphProto,
  results: List[graph_tuple_database.GraphTuple],
  ctx: progress.ProgressBarContext,
) -> None:
  """Create annotated graphs from a single protocol buffer input."""
  annotator = GetDataFlowGraphAnnotator(analysis)

  # De-serialize the binary protocol buffer.
  proto = programl_pb2.ProgramGraph()
  proto.ParseFromString(program_graph.serialized_proto)

  # Convert the proto into a networkx graph.
  graph = programl.ProgramGraphToNetworkX(proto)

  try:
    # Create annotated graphs.
    annotated_graphs = annotator.MakeAnnotated(
      graph, FLAGS.max_instances_per_graph
    )
    # Create GraphTuple database instances from annotated graphs and append
    # them to the output results list.
    results += [
      graph_tuple_database.GraphTuple.CreateFromDataFlowAnnotatedGraph(
        annotated_graph, ir_id=program_graph.ir_id
      )
      for annotated_graph in annotated_graphs
    ]
  except Exception as e:
    results.append(
      graph_tuple_database.GraphTuple.CreateEmpty(ir_id=program_graph.ir_id)
    )
    _, _, tb = sys.exc_info()
    tb = traceback.extract_tb(tb, 2)
    filename, line_number, function_name, *_ = tb[-1]
    filename = pathlib.Path(filename).name
    ctx.Error(
      "Failed to annotate graph for ProgramGraph.ir_id=%d: %s "
      "(%s:%s:%s() -> %s)",
      program_graph.ir_id,
      e,
      filename,
      line_number,
      function_name,
      type(e).__name__,
    )


class AnnotationResult(NamedTuple):
  """The result of running ProcessProgramGraphs() on a list of protos."""

  runtime: float
  proto_count: int
  graph_tuples: List[graph_tuple_database.GraphTuple]


def ProcessProgramGraphs(packed_args) -> AnnotationResult:
  start_time = time.time()

  # Unpack the args generated by ProcessProgramGraphsArgsGenerator().
  # Index into the tuple rather than arg unpacking so that we can assign
  # type annotations.
  worker_id: str = f"{packed_args[0]:06d}"
  analysis: str = packed_args[1]
  program_graphs: List[ProgramGraphProto] = packed_args[2]
  ctx: progress.ProgressBarContext = packed_args[3]

  proto_count = 0
  annotated_graphs = []

  with ctx.Profile(
    2,
    lambda t: (
      f"[worker {worker_id} processed {len(annotated_graphs)} protos "
      f"({len(annotated_graphs)} graphs)"
    ),
  ):
    for i, program_graph in enumerate(program_graphs):
      proto_count += 1
      # Run a separate process for each graph. This incurs a signficant overhead
      # for processing graphs, but (as far as I can tell) is the only way of
      # enforcing a per-proto timeout for the annotation.
      process = multiprocessing.Process(
        target=ProcessOneProto,
        args=(analysis, program_graph, annotated_graphs, ctx),
      )
      process.start()

      with ctx.Profile(
        3, f"[worker {worker_id}] Processed proto [{i+1}/{len(program_graphs)}]"
      ):
        process.join(FLAGS.annotator_timeout)
      if process.is_alive():
        ctx.Error(
          f"Failed to annotate proto {program_graph.ir_id} in "
          f"{FLAGS.annotator_timeout} seconds"
        )
        process.terminate()
        process.join()

  return AnnotationResult(
    runtime=time.time() - start_time,
    proto_count=proto_count,
    graph_tuples=annotated_graphs,
  )


class DatasetGenerator(progress.Progress):
  """Worker thread for dataset."""

  def __init__(
    self,
    input_db: unlabelled_graph_database.Database,
    analysis: str,
    output_db: graph_tuple_database.Database,
  ):
    self.analysis = analysis
    self.output_db = output_db

    # Create an annotator now to crash-early if the analysis name is invalid.
    GetDataFlowGraphAnnotator(analysis)

    # Get the graphs that have already been processed.
    with output_db.Session() as out_session:
      already_done_max, already_done_count = out_session.query(
        sql.func.max(graph_tuple_database.GraphTuple.ir_id),
        sql.func.count(
          sql.func.distinct(graph_tuple_database.GraphTuple.ir_id)
        ),
      ).one()
      already_done_max = already_done_max or -1

    # Get the total number of graphs to process, and the IDs of the graphs to
    # process.
    with input_db.Session() as in_session:
      total_graph_count = in_session.query(
        sql.func.count(unlabelled_graph_database.ProgramGraph.ir_id)
      ).scalar()
      ids_and_sizes_to_do = (
        in_session.query(
          unlabelled_graph_database.ProgramGraph.ir_id,
          unlabelled_graph_database.ProgramGraph.serialized_proto_size,
        )
        .filter(unlabelled_graph_database.ProgramGraph.ir_id > already_done_max)
        .order_by(unlabelled_graph_database.ProgramGraph.ir_id)
      )
      # Optionally limit the number of IDs to process.
      if FLAGS.max_instances:
        ids_and_sizes_to_do = ids_and_sizes_to_do.limit(FLAGS.max_instances)
      ids_and_sizes_to_do = [
        (row.ir_id, row.serialized_proto_size) for row in ids_and_sizes_to_do
      ]

    # Sanity check.
    if not FLAGS.max_instances:
      if len(ids_and_sizes_to_do) + already_done_count != total_graph_count:
        raise OSError(
          "ids_to_do(%s) + already_done(%s) != total_rows(%s)",
          len(ids_and_sizes_to_do),
          already_done_count,
          total_graph_count,
        )

    with output_db.Session(commit=True) as out_session:
      out_session.add(
        unlabelled_graph_database.Meta.Create(
          key="Graph counts", value=(already_done_count, total_graph_count)
        )
      )
    app.Log(
      1,
      "Selected %s of %s to process",
      humanize.Commas(len(ids_and_sizes_to_do)),
      humanize.Plural(total_graph_count, "unlabelled graph"),
    )

    super(DatasetGenerator, self).__init__(
      name=analysis, i=already_done_count, n=total_graph_count, unit="protos"
    )

    self.graph_reader = ppar.ThreadedIterator(
      BatchedProtoReader(
        input_db,
        ids_and_sizes_to_do,
        FLAGS.proto_batch_mb * 1024 * 1024,
        self.ctx.ToProgressContext(),
      ),
      max_queue_size=FLAGS.max_reader_queue_size,
    )

  def RunWithPool(
    self, pool: multiprocessing.Pool, writer: sqlutil.BufferedDatabaseWriter
  ):
    def ProcessProgramGraphsArgsGenerator(graph_reader):
      """Generate packed arguments for a multiprocessing worker."""
      for i, graph_batch in enumerate(graph_reader):
        yield (i, self.analysis, graph_batch, self.ctx.ToProgressContext())

    # Have a thread generating inputs, a multiprocessing pool processing them,
    # and another thread writing their results to the database.
    worker_args = ProcessProgramGraphsArgsGenerator(self.graph_reader)
    # Process the inputs using an iterator, and enforce that the results
    # arrive *in order*, as we process the input database in-order.
    workers = pool.imap(
      ProcessProgramGraphs, worker_args, chunksize=FLAGS.chunk_size
    )

    for elapsed_time, proto_count, graph_tuples in workers:
      self.ctx.i += proto_count
      # Record the generated annotated graphs.
      tuple_sizes = [t.pickled_graph_tuple_size for t in graph_tuples]
      writer.AddMany(graph_tuples, sizes=tuple_sizes)
    if writer.error_count:
      raise OSError("Database writer had errors")

    # Sanity check the number of generated program graphs.
    # If --max_instances is set, this means the script will fail unless the
    # entire dataset has been processed.
    if self.ctx.i != self.ctx.n:
      raise OSError(
        "unlabelled_graph_count(%s) != exported_count(%s)",
        self.ctx.n,
        self.ctx.i,
      )
    with self.output_db.Session() as out_session:
      annotated_graph_count = out_session.query(
        sql.func.count(sql.func.distinct(graph_tuple_database.GraphTuple.ir_id))
      ).scalar()
    if annotated_graph_count != self.ctx.n:
      raise OSError(
        "unlabelled_graph_count(%s) != annotated_graph_count(%s)",
        self.ctx.n,
        annotated_graph_count,
      )

  def Run(self):
    """Run the dataset generation."""
    pool = ppar.UnsafeNonDaemonPool(
      processes=FLAGS.nproc, maxtasksperchild=FLAGS.max_tasks_per_worker
    )
    try:
      with sqlutil.BufferedDatabaseWriter(
        self.output_db,
        max_buffer_size=FLAGS.write_buffer_mb * 1024 * 1024,
        max_buffer_length=FLAGS.write_buffer_length,
        log_level=1,
        ctx=self.ctx.ToProgressContext(),
      ) as writer:
        self.DoExport(pool, writer)
    finally:
      # Make sure to tidy up after ourselves.
      pool.close()
      pool.join()


def main():
  """Main entry point."""
  if not FLAGS.proto_db:
    raise app.UsageError("--proto_db required")
  if not FLAGS.graph_db:
    raise app.UsageError("--graph_db required")

  input_db = FLAGS.proto_db()
  output_db = FLAGS.graph_db()

  progress.Run(DatasetGenerator(input_db, FLAGS.analysis, output_db))


if __name__ == "__main__":
  app.Run(main)
