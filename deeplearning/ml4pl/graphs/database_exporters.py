"""A module which defines a base class for implementing database exporters."""
import multiprocessing
import pathlib
import random
import tempfile
import time
import typing

from labm8 import app
from labm8 import fs
from labm8 import humanize
from labm8 import labtypes
from labm8 import prof
from labm8 import sqlutil
from labm8 import system

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.bytecode import splitters
from deeplearning.ml4pl.graphs import graph_database

FLAGS = app.FLAGS

app.DEFINE_integer('nproc', multiprocessing.cpu_count(),
                   'Enable multiprocessing for database job workers.')
app.DEFINE_integer(
    'batch_size', 32,
    'The number of bytecodes to process in-memory before writing'
    'to database.')
app.DEFINE_integer(
    'seed', 0xCEC,
    'The random seed value to use when shuffling graph statements when '
    'selecting the root statement.')


class DatabaseExporterBase(object):
  """Base class for implementing parallelized database workers."""

  def GetProcessInputs(
      self) -> typing.Callable[[sqlutil.Database, typing.List[int]], typing.
                               Optional[typing.Any]]:
    """A method which returns a function that accepts as input a database
    and an index into a table in the database and yields a list of zero or more
    graph metas.
    """
    raise NotImplementedError("abstract class")

  def Export(self, input_db: sqlutil.Database,
             output_dbs: typing.List[graph_database.Database],
             pool: multiprocessing.Pool, batch_size: int):
    raise NotImplementedError("abstract class")

  def __call__(self,
               input_db: sqlutil.Database,
               output_dbs: typing.List[graph_database.Database],
               pool: typing.Optional[multiprocessing.Pool] = None,
               batch_size: typing.Optional[int] = None):
    pool = pool or multiprocessing.Pool(processes=1)
    batch_size = batch_size or FLAGS.batch_size

    with tempfile.TemporaryDirectory() as d:
      # Temporarily redirect logs to a file, which we will later import into the
      # database's meta table.
      FLAGS.alsologtostderr = True
      app.LogToDirectory(d, 'log')

      try:
        app.Log(1, 'Seeding export with %s', FLAGS.seed)
        random.seed(FLAGS.seed)
        self.Export(input_db, output_dbs, pool, batch_size)
      finally:
        # Add a 'log' entry to the export database.
        log = fs.Read(pathlib.Path(d) / 'log.INFO')
        for output_db in output_dbs:
          with output_db.Session(commit=True) as s:
            run_id = (f"{time.strftime('%Y%m%dT%H%M%S')}@" f"{system.HOSTNAME}")
            s.add(graph_database.Meta(key=f'log:{run_id}', value=log))


class BytecodeDatabaseExporterBase(DatabaseExporterBase):
  """Base class for implementing parallelized LLVM bytecode database workers."""

  def Export(self, input_db: sqlutil.Database,
             output_dbs: typing.List[graph_database.Database],
             pool: typing.Optional[multiprocessing.Pool],
             batch_size: typing.Optional[int]):
    if len(output_dbs) != 1:
      raise TypeError("Bytecode exporters only support a single output "
                      "database.")

    group_to_ids_map = splitters.GetGroupsFromFlags(input_db)

    start_time = time.time()
    group_to_graph_count_map = dict()
    # Export from each group in turn.
    for group, bytecode_ids in group_to_ids_map.items():
      group_start_time = time.time()
      exported_graph_count = self.ExportGroup(input_db, output_dbs[0], group,
                                              bytecode_ids, pool, batch_size)
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

  def ExportGroup(self, input_db: sqlutil.Database,
                  output_db: graph_database.Database, group: str,
                  bytecode_ids: typing.List[int], pool: multiprocessing.Pool,
                  batch_size: int) -> int:
    """Export the given group.

    Args:
      group: The name of the group.
      bytecode_ids: The bytecodes to export.

    Returns:
      The number of graphs exported.
    """
    start_time = time.time()
    exported_count = 0

    # Ignore bytecodes that we have already exported.
    with output_db.Session() as session:
      query = session.query(graph_database.GraphMeta.bytecode_id) \
        .filter(graph_database.GraphMeta.bytecode_id.in_(bytecode_ids))
      already_done = set([r.bytecode_id for r in query])
      app.Log(1, 'Skipping %s previously-exported bytecodes',
              humanize.Commas(len(already_done)))
      bytecode_ids = [b for b in bytecode_ids if b not in already_done]

    # Process bytecodes in a random order.
    random.shuffle(bytecode_ids)

    job_processor = self.GetProcessInputs()
    bytecode_id_chunks = labtypes.Chunkify(bytecode_ids, batch_size)
    jobs = [(job_processor, input_db.url, bytecode_ids_chunk)
            for bytecode_ids_chunk in bytecode_id_chunks]
    app.Log(1, "Divided %s %s bytecode chunks into %s jobs",
            humanize.Commas(len(bytecode_ids)), group, len(jobs))

    if FLAGS.nproc > 1:
      workers = pool.imap_unordered(_BytecodeWorker, jobs)
    else:
      workers = (_BytecodeWorker(job) for job in jobs)

    job_count = 0
    for graph_metas in workers:
      exported_count += len(graph_metas)
      job_count += 1
      app.Log(
          1,
          'Created %s `%s` graphs from %s inputs (%.2f%%, %.2f graphs/second)',
          humanize.Commas(len(graph_metas)), group, len(bytecode_ids),
          (job_count / len(jobs)) * 100,
          exported_count / (time.time() - start_time))

      # Set the GraphMeta.group column.
      for graph in graph_metas:
        graph.group = group

      if graph_metas:
        sqlutil.ResilientAddManyAndCommit(output_db, graph_metas)

    return exported_count


def _BytecodeWorker(packed_args):
  """A bytecode processor worker. If --nproc is set,
  this is called in a worker process.
  """
  job_processor, bytecode_db_url, bytecode_ids = packed_args
  with prof.Profile(lambda t: (f"Processed {humanize.Commas(len(bytecode_ids))}"
                               f" bytecodes ({len(bytecode_ids) / t:.2f} "
                               f"bytecodes/sec)")):
    bytecode_db = bytecode_database.Database(bytecode_db_url)
    return job_processor(bytecode_db, bytecode_ids)


class GraphDatabaseExporterBase(DatabaseExporterBase):
  """Base class for implementing parallelized graph database workers.

  This supports multiple output databases, where each output database has a
  one-to-many relationship to the input database, identified by unique bytecode
  IDs.
  """

  def Export(self, input_db: sqlutil.Database,
             output_dbs: typing.List[graph_database.Database],
             pool: typing.Optional[multiprocessing.Pool],
             batch_size: typing.Optional[int]):
    if len(output_dbs) != 1:
      raise TypeError
    output_db = output_dbs[0]

    start_time = time.time()
    exported_graph_count = 0

    # Get the bytecode IDs of the graphs to export.
    with input_db.Session() as session:
      query = session.query(graph_database.GraphMeta.bytecode_id)
      bytecode_ids = set([row.bytecode_id for row in query])

    # Ignore bytecodes that we have already exported.
    with output_db.Session() as session:
      query = session.query(graph_database.GraphMeta.bytecode_id) \
        .filter(graph_database.GraphMeta.bytecode_id.in_(bytecode_ids))
      already_done = set([row.bytecode_id for row in query])
      app.Log(1, 'Skipping %s previously-exported bytecodes',
              humanize.Commas(len(already_done)))
      bytecode_ids = [b for b in bytecode_ids if b not in already_done]

    # Process bytecodes in a random order.
    random.shuffle(bytecode_ids)

    job_processor = self.GetProcessInputs()
    bytecode_id_chunks = labtypes.Chunkify(bytecode_ids, FLAGS.batch_size)
    jobs = [(job_processor, input_db.url, bytecode_ids_chunk)
            for bytecode_ids_chunk in bytecode_id_chunks]
    app.Log(1, "Divided %s bytecode chunks into %s jobs",
            humanize.Commas(len(bytecode_ids)), len(jobs))

    if FLAGS.nproc:
      workers = pool.imap_unordered(_GraphWorker, jobs)
    else:
      workers = (_GraphWorker(job) for job in jobs)

    job_count = 0
    for graph_metas in workers:
      exported_graph_count += len(graph_metas)
      job_count += 1
      app.Log(
          1, 'Created %s graphs at %.2f graphs/sec. %.2f%% of %s bytecodes '
          'processed', humanize.Commas(len(graph_metas)),
          exported_graph_count / (time.time() - start_time),
          (job_count / len(jobs)) * 100,
          humanize.DecimalPrefix(len(bytecode_ids), ''))

      if graph_metas:
        sqlutil.ResilientAddManyAndCommit(output_db, graph_metas)

    elapsed_time = time.time() - start_time
    app.Log(
        1, 'Exported %s graphs from %s input graphs in %s '
        '(%.2f graphs / second)', humanize.Commas(exported_graph_count),
        humanize.Commas(len(bytecode_ids)), humanize.Duration(elapsed_time),
        exported_graph_count / elapsed_time)
    return exported_graph_count


def _GraphWorker(packed_args):
  """A graph processor worker. If --nproc is set,
  this is called in a worker process.
  """
  job_processor, graph_db_url, bytecode_ids = packed_args
  with prof.Profile(lambda t: (f"Processed {humanize.Commas(len(bytecode_ids))}"
                               f" input graphs ({len(bytecode_ids) / t:.2f} "
                               f"input graphs/sec)")):
    graph_db = graph_database.Database(graph_db_url)
    return job_processor(graph_db, bytecode_ids)
