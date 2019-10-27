"""A module which defines a base class for implementing database exporters."""
import math
import multiprocessing
import time
import typing

from labm8 import app
from labm8 import humanize
from labm8 import labtypes
from labm8 import ppar
from labm8 import prof
from labm8 import sqlutil

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import graph_database

FLAGS = app.FLAGS

app.DEFINE_boolean('multiprocess_database_exporters', True,
                   'Enable multiprocessing for database job workers.')
app.DEFINE_integer(
    'database_exporter_batch_size', 1024,
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

  def GetMakeExportJob(
      self) -> typing.Callable[[bytecode_database.Database.
                                SessionType, int], typing.Optional[typing.Any]]:
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
          1, 'Exported %s %s graphs from %s bytecodes in %s '
          '(%.2f graphs / second)', humanize.Commas(exported_graph_count),
          group, humanize.Commas(len(bytecode_ids)),
          humanize.Duration(elapsed_time), exported_graph_count / elapsed_time)
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
    """Export the given groups."""
    start_time = time.time()
    exported_count = 0

    make_job = self.GetMakeExportJob()
    job_processor = self.GetProcessJobFunction()

    chunksize = min(
        max(math.ceil(len(bytecode_ids) / self.pool._processes), 8),
        self.batch_size)

    bytecode_id_chunks = labtypes.Chunkify(bytecode_ids, chunksize)
    jobs = [(self.bytecode_db.url, bytecode_ids_chunk, make_job, job_processor)
            for bytecode_ids_chunk in bytecode_id_chunks]
    app.Log(1, "Divided %s %s bytecode chunks into %s jobs",
            humanize.Commas(len(bytecode_ids)), group, len(jobs))

    if FLAGS.multiprocess_database_exporters:
      workers = self.pool.imap_unordered(_Worker, jobs)
    else:
      workers = (_Worker(job) for job in jobs)

    job_count = 0
    with sqlutil.BufferedDatabaseWriter(self.graph_db).Session() as writer:
      for graph_metas in workers:
        exported_count += len(graph_metas)
        job_count += 1
        app.Log(1, 'Created %s %s graphs (%.2f%%, %.2f graphs/second)',
                humanize.Commas(len(graph_metas)), group,
                (job_count / len(jobs)) * 100,
                exported_count / (time.time() - start_time))

        # Set the GraphMeta.group column.
        for graph in graph_metas:
          graph.group = group
        writer.AddMany(graph_metas)

    return exported_count


def _Worker(packed_args):
  """A bytecode processor worker. If --multiprocess_database_exporters is set,
  this is called in a worker process.
  """
  bytecode_db_url, bytecode_ids, make_job, job_processor = packed_args

  with prof.Profile(lambda t: (f"Created {len(jobs)} jobs from "
                               f"{len(bytecode_ids)} bytecodes "
                               f"({len(bytecode_ids) / t:.2f} "
                               "bytecodes/sec)")):
    bytecode_db = bytecode_database.Database(bytecode_db_url)
    with bytecode_db.Session() as session:
      jobs = [make_job(session, bytecode_id) for bytecode_id in bytecode_ids]
    # Filter the failed jobs.
    jobs = [j for j in jobs if j]

  graph_metas = []
  for job in jobs:
    graph_metas += job_processor(job)

  return graph_metas
