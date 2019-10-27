"""A module which defines a base class for implementing database exporters."""
import multiprocessing
import time
import typing

from labm8 import app
from labm8 import humanize
from labm8 import labtypes
from labm8 import sqlutil

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import graph_database

FLAGS = app.FLAGS

app.DEFINE_boolean('multiprocess_database_exporters', True,
                   'Enable multiprocessing for database job workers.')
app.DEFINE_integer(
    'database_exporter_batch_size', 10000,
    'The number of bytecodes to process in-memory before writing'
    'to database.')


class BytecodeDatabaseExporterBase(object):
  """Abstract base class for implementing parallelized LLVM bytecode workers."""

  def __init__(self,
               bytecode_db: bytecode_database.Database,
               graph_db: graph_database.Database,
               pool: typing.Optional[multiprocessing.Pool] = None,
               batch_size: typing.Optional[int] = None):
    self.bytecode_db = bytecode_db
    self.graph_db = graph_db
    self.pool = pool or multiprocessing.Pool()
    self.batch_size = batch_size or FLAGS.database_exporter_batch_size

  def MakeExportJob(self, session: bytecode_database.Database.SessionType,
                    bytecode_id: int) -> typing.Optional[typing.Any]:
    raise NotImplementedError("abstract class")

  def GetProcessJobFunction(
      self
  ) -> typing.Callable[[typing.Any], typing.List[graph_database.GraphMeta]]:
    raise NotImplementedError("abstract class")

  def ExportGroups(self,
                   group_to_ids_map: typing.Dict[str, typing.List[int]]) -> int:
    start_time = time.time()
    group_to_graph_count_map = dict()
    # Export from each group in turn.
    for group, bytecode_ids in group_to_ids_map.items():
      group_start_time = time.time()
      exported_graph_count = self.ExportGroup(group, bytecode_ids)
      elapsed_time = time.time() - group_start_time
      app.Log(
          1, 'Exported %s graphs from %s bytecodes in %s '
          '(%.2f graphs / second)', humanize.Commas(exported_graph_count),
          humanize.Commas(len(bytecode_ids)), humanize.Duration(elapsed_time),
          exported_graph_count / elapsed_time)
      group_to_graph_count_map[group] = exported_graph_count

    total_count = sum(group_to_graph_count_map.values())
    elapsed_time = time.time() - start_time

    group_str = ', '.join([
        f'{humanize.Commas(count)} {group}'
        for group, count in sorted(group_to_graph_count_map.items())
    ])
    app.Log(1, 'Exported %s graphs (%s) in %s (%.2f graphs / second)',
            humanize.Commas(total_count), group_str,
            humanize.Duration(elapsed_time), total_count / elapsed_time)

    return total_count

  def ExportGroup(self, group: str, bytecode_ids: typing.List[int]):
    exported_count = 0

    chunksize = max(self.batch_size // 16, 8)
    job_processor = self.GetProcessJobFunction()

    for i, chunk in enumerate(labtypes.Chunkify(bytecode_ids, self.batch_size)):
      app.Log(1, 'Processing %s-%s of %s bytecodes (%.2f%%)',
              i * self.batch_size, i * self.batch_size + len(chunk),
              humanize.Commas(len(bytecode_ids)),
              ((i * self.batch_size) / len(bytecode_ids)) * 100)
      # Run the database queries from the master thread to produce
      # jobs.
      with self.bytecode_db.Session() as s:
        jobs = [self.MakeExportJob(s, bytecode_id) for bytecode_id in chunk]
      # Filter the failed jobs.
      jobs = [j for j in jobs if j]

      # Process jobs in parallel.
      graph_metas = []
      if FLAGS.multiprocess_database_exporters:
        workers = self.pool.imap_unordered(job_processor,
                                           jobs,
                                           chunksize=chunksize)
      else:
        workers = (job_processor(job) for job in jobs)
      for graphs_chunk in workers:
        graph_metas += graphs_chunk

      exported_count += len(graph_metas)
      # Set the GraphMeta.group column.
      for graph in graph_metas:
        graph.group = group
      sqlutil.ResilientAddManyAndCommit(self.graph_db, graph_metas)

    return exported_count
