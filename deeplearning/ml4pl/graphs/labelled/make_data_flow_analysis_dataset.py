"""This module prepares datasets for data flow analyses."""
import math
import multiprocessing
import pathlib
import random
import sys
import time
import traceback
import typing

import numpy as np
import sqlalchemy as sql
from labm8 import app
from labm8 import humanize
from labm8 import prof
from labm8 import sqlutil

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import database_exporters
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled.alias_set import alias_set
from deeplearning.ml4pl.graphs.labelled.datadep import data_dependence
from deeplearning.ml4pl.graphs.labelled.domtree import dominator_tree
from deeplearning.ml4pl.graphs.labelled.liveness import liveness
from deeplearning.ml4pl.graphs.labelled.polyhedra import polyhedra
from deeplearning.ml4pl.graphs.labelled.reachability import reachability
from deeplearning.ml4pl.graphs.labelled.subexpressions import subexpressions

app.DEFINE_database(
    'input_graphs_db',
    graph_database.Database,
    None,
    'URL of database to read unlabelled networkx graphs from.',
    must_exist=True)
app.DEFINE_database(
    'bytecode_db',
    bytecode_database.Database,
    None, 'URL of database to read bytecode from. Only required when '
    'analysis requires bytecode.',
    must_exist=True)
app.DEFINE_list(
    'outputs', None,
    "A list of outputs to generate, where each element in the list"
    "is in the form '<analysis>:<db_url>' and names the analysis to"
    "run and URL of the database to write to, respectively.")
app.DEFINE_integer(
    'max_instances_per_graph', 10,
    'The maximum number of instances to produce from a single input graph. '
    'For a CDFG with `n` statements, `n` instances can be '
    'produced by changing the root statement for analyses.')
app.DEFINE_string('order_by', 'frequency', 'The order to process in')
app.DEFINE_boolean('error', False, 'If true, crash on export error.')

FLAGS = app.FLAGS


class GraphAnnotator(typing.NamedTuple):
  """A named tuple describing a graph annotator, which is a function that
  accepts produces labelled graphs based on input graphs/bytecodes."""
  # The human-interpretable name of the analysis.
  name: str

  # The function that produces labelled GraphMeta instances.
  function: typing.Callable[[typing.Any], typing.List[graph_database.GraphMeta]]

  # If true, a list of networkx graphs is passed to `function` as named
  # argument 'graphs'.
  requires_graphs: bool = False

  # If true, a list of bytecodes is passed to `function` as named argument
  # 'bytecodes'.
  requires_bytecodes: bool = False


def GetAnnotatedGraphGenerators(
    *analysis_names: typing.Iterable[str]) -> typing.List[GraphAnnotator]:
  """Return the graph annotators for the requested analyses. If no analyses are
  provided, all annotators are returned."""
  annotators: typing.List[GraphAnnotator] = []

  if analysis_names:  # Strip duplicates from requested analyses.
    analysis_names = set(analysis_names)

  def AnalysisIsRequested(analysis_name: str, analysis_names: typing.Set[str]):
    """Determine if the given analysis has been requested."""
    if analysis_name in analysis_names:
      analysis_names.remove(analysis_name)
      return True
    else:
      return False

  if AnalysisIsRequested('reachability', analysis_names):
    annotators.append(
        GraphAnnotator(
            name='reachability',
            requires_graphs=True,
            function=reachability.MakeReachabilityGraphs))

  if AnalysisIsRequested('domtree', analysis_names):
    annotators.append(
        GraphAnnotator(
            name='domtree',
            requires_graphs=True,
            function=dominator_tree.MakeDominatorTreeGraphs))

  if AnalysisIsRequested('datadep', analysis_names):
    annotators.append(
        GraphAnnotator(
            name='datadep',
            requires_graphs=True,
            function=data_dependence.MakeDataDependencyGraphs))

  if AnalysisIsRequested('liveness', analysis_names):
    annotators.append(
        GraphAnnotator(
            name='liveness',
            requires_graphs=True,
            function=liveness.MakeLivenessGraphs))

  if AnalysisIsRequested('subexpressions', analysis_names):
    annotators.append(
        GraphAnnotator(
            name='subexpressions',
            requires_graphs=True,
            function=subexpressions.MakeSubexpressionsGraphs))

  if AnalysisIsRequested('alias_sets', analysis_names):
    annotators.append(
        GraphAnnotator(
            name='alias_sets',
            requires_graphs=True,
            requires_bytecodes=True,
            function=alias_set.MakeAliasSetGraphs))

  if AnalysisIsRequested('polyhedra', analysis_names):
    annotators.append(
        GraphAnnotator(
            name='polyhedra',
            requires_graphs=False,
            requires_bytecodes=True,
            function=polyhedra.MakePolyhedralGraphs))

  if analysis_names:
    raise app.UsageError(f"Unknown analyses {analysis_names}")

  return annotators


class Output(typing.NamedTuple):
  """An output to generate."""
  # The graph annotator.
  annotator: GraphAnnotator
  # The database to write annotated graphs to.
  db: graph_database.Database


def GetOutputsFromStrings(outputs: typing.List[str]) -> typing.List[Output]:
  annotators = GetAnnotatedGraphGenerators(*[s.split(':')[0] for s in outputs])
  db_urls = [':'.join(s.split(':')[1:]) for s in outputs]
  dbs = [graph_database.Database(url) for url in db_urls]

  if not annotators:
    raise app.UsageError("At least one output is required")

  return [
      Output(annotator=annotator, db=db)
      for annotator, db in zip(annotators, dbs)
  ]


def GetAllBytecodeIds(db: graph_database.Database) -> typing.Set[int]:
  """Read all unique bytecode IDs from the database as a set."""
  with db.Session() as session:
    query = session.query(
        graph_database.GraphMeta.bytecode_id.distinct().label('bytecode_id'))
    all_ids = set([row.bytecode_id for row in query])
  return all_ids


def GetBytecodeIdsToProcess(
    input_ids: typing.Set[int],
    output_dbs: typing.List[graph_database.Database],
    batch_size: int) -> typing.Tuple[typing.List[int], int]:
  """Get the bytecode IDs to process.

  This returns a tuple of <all_bytecode_ids,bytecodes_by_output>, where
  all_bytecode_ids is an array of shape (min(n,10000),?) and includes the
  total set of unique bytecodes to be processed. bytecodes_by_output is a
  matrix of shape (num_outputs,min(n,10000)), where each row corresponds with
  an output_db. Bytecodes that already exist in an output database are marked
  with zeros. So for eah row in bytecodes_by_output, the bytecodes that need
  processing for a particular output are row[np.nonzero(row)].
  """
  with prof.Profile(lambda t: (
      "Read the "
      f"{humanize.Commas(len(all_bytecodes_to_process))} jobs "
      f"to process")):
    ids_todo_by_output = [
        input_ids - GetAllBytecodeIds(output_db) for output_db in output_dbs
    ]

    # Flatten the list of all bytecodes that haven't been processed. This list
    # includes duplicates for bytecodes that need processing by multiple
    # outputs. This is intentional, allowing us to sort bytecodes by frequency.
    all_bytecodes_to_process: typing.List[str] = []
    for ids in ids_todo_by_output:
      all_bytecodes_to_process.extend(list(ids))

  with prof.Profile(lambda t: (
      f"Selected {humanize.Commas(len(bytecodes_subset))} of "
      f"{humanize.Commas(len(frequency_table))} bytecodes to "
      "process with "
      f"{humanize.Commas(todo_by_output[np.nonzero(todo_by_output)].size)} annotations"
  )):
    if FLAGS.order_by == 'random':
      bytecodes_to_process = np.array(
          list(set(all_bytecodes_to_process)), dtype=np.int32)
      frequency_table = bytecodes_to_process  # Used in prof.Profile() callback.
      random.shuffle(bytecodes_to_process)
      bytecodes_subset = bytecodes_to_process[:batch_size]
    elif FLAGS.order_by == 'frequency':
      # Create a frequency table how for many times each unprocessed bytecode
      # occurs.
      frequency_table = np.vstack(
          np.unique(all_bytecodes_to_process, return_counts=True)).T
      # Sort the frequency table by count so that most frequently unprocessed
      # bytecodes occur *at the end* of the list.
      sorted_frequency_table = frequency_table[frequency_table[:, 1].argsort()]
      bytecodes_subset = sorted_frequency_table[-batch_size:, 0]
    elif FLAGS.order_by == 'reverse_frequency':
      # Create a frequency table how for many times each unprocessed bytecode
      # occurs.
      frequency_table = np.vstack(
          np.unique(all_bytecodes_to_process, return_counts=True)).T
      # Sort the frequency table by count so that most frequently unprocessed
      # bytecodes occur *at the end* of the list.
      sorted_frequency_table = frequency_table[frequency_table[:, 1].argsort()]
      bytecodes_subset = sorted_frequency_table[batch_size:, 0]
    else:
      raise app.UsageError("Unknown `order_by` option.")

    # Produce the zero-d matrix of bytecodes that need processing for each
    # output.
    todo_by_output = []
    for ids_for_output in ids_todo_by_output:
      todo_by_output.append(
          [x if x in ids_for_output else 0 for x in bytecodes_subset])
    todo_by_output = np.vstack(todo_by_output)

  return bytecodes_subset, todo_by_output


def ResilientAddUnique(db: graph_database.Database,
                       graph_metas: typing.List[graph_database.GraphMeta],
                       annotator_name: str) -> None:
  """Attempt to commit all graph metas to the database.

  This function adds graph metas to the database, provided they do not already
  exist.

  Args:
    db: The database to add the objects to.
    graph_metas: A sequence of graph metas to commit.

  Returns:
    The number of graphs added.
  """
  if not graph_metas:
    return 0

  try:
    bytecode_ids = {g.bytecode_id for g in graph_metas}
    with db.Session(commit=True) as session:
      # Get the bytecodes which have already been imported into the database and
      # commit only the new ones. This is prevention against multiple-versions
      # of the graph being added when there are parallel importers.
      query = session.query(
          graph_database.GraphMeta.bytecode_id.distinct().label('bytecode_id')) \
        .filter(graph_database.GraphMeta.bytecode_id.in_(bytecode_ids))
      already_done_ids = {row.bytecode_id for row in query}
      graph_metas_to_commit = [
          g for g in graph_metas if g.bytecode_id not in already_done_ids
      ]
      if len(graph_metas_to_commit) < len(graph_metas):
        app.Warning(
            'Ignoring %s %s graph metas that already exist in the database',
            len(graph_metas) - len(graph_metas_to_commit), annotator_name)
      session.add_all(graph_metas_to_commit)
      return len(graph_metas_to_commit)
  except sql.exc.SQLAlchemyError as e:
    app.Log(
        1,
        'Caught error while committing %d %s graphs: %s',
        len(graph_metas),
        annotator_name,
        e,
    )

    # Divide and conquer. If we're committing only a single object, then a
    # failure to commit it means that we can do nothing other than return it.
    # Else, divide the mapped objects in half and attempt to commit as many of
    # them as possible.
    if len(graph_metas) == 1:
      return
    else:
      mid = int(len(graph_metas) / 2)
      left = graph_metas[:mid]
      right = graph_metas[mid:]
      return (ResilientAddUnique(db, left, annotator_name) + ResilientAddUnique(
          db, right, annotator_name))


class DataFlowAnalysisGraphExporter(database_exporters.DatabaseExporterBase):
  """Create and add graph annotator annotations."""

  def __init__(self, outputs: typing.List[Output]):
    super(DataFlowAnalysisGraphExporter, self).__init__()
    self.outputs = outputs

  def Export(self, input_db: sqlutil.Database,
             output_dbs: typing.List[graph_database.Database],
             pool: multiprocessing.Pool, batch_size: int) -> None:
    del output_dbs  # Unused, self.outputs holds the output databases.

    start_time = time.time()

    # Read all of the bytecode IDs from the input database.
    with prof.Profile(lambda t: (
        f"Read {humanize.Commas(len(input_ids))} input "
        "bytecode IDs")):
      input_ids = GetAllBytecodeIds(input_db)

    all_exported_graph_count = 0
    exported_graph_count = self._Export(input_db, input_ids, pool, batch_size)
    while exported_graph_count:
      all_exported_graph_count += exported_graph_count
      exported_graph_count = self._Export(input_db, input_ids, pool, batch_size)

    elapsed_time = time.time() - start_time
    app.Log(1, 'Exported %s graphs in %s '
            '(%.2f graphs / second)', humanize.Commas(exported_graph_count),
            humanize.Duration(elapsed_time),
            exported_graph_count / elapsed_time)

  def _Export(self, input_db: sqlutil.Database, input_ids: typing.Set[int],
              pool: multiprocessing.Pool, batch_size: int) -> int:
    start_time = time.time()
    exported_graph_count = 0

    bytecodes_to_process, bytecodes_to_process_by_output = GetBytecodeIdsToProcess(
        input_ids, [output.db for output in self.outputs], batch_size)
    if not bytecodes_to_process.size:
      return 0

    annotators: typing.List[GraphAnnotator] = [
        x.annotator for x in self.outputs
    ]

    # Break the bytecodes to process into chunks.
    bytecode_id_chunks = np.split(
        bytecodes_to_process_by_output,
        list(
            range(0, len(bytecodes_to_process_by_output[0]),
                  max(batch_size // FLAGS.nproc, 1)))[1:],
        axis=1)
    jobs = list((input_db.url, annotators, bytecode_ids_chunk)
                for bytecode_ids_chunk in bytecode_id_chunks)
    app.Log(1, "Divided %s bytecodes into %s %s-bytecode jobs",
            humanize.Commas(len(bytecodes_to_process)), len(bytecode_id_chunks),
            batch_size)

    # Split the jobs into smaller chunks so that imap_unordered doesn't outrun
    # the database writer by too much.
    # for jobs_chunk in iter(
    #     lambda: list(itertools.islice(jobs, FLAGS.nproc * 2)), []):
    # app.Log(1, 'Starting chunk of %s jobs', FLAGS.nproc * 2)
    if FLAGS.nproc > 1:
      workers = pool.imap_unordered(_Worker, jobs)
    else:
      workers = (_Worker(job) for job in jobs)

    job_count = 0
    for annotator, graph_metas_by_output in zip(annotators, workers):
      job_count += 1
      app.Log(
          1, 'Created %s graphs at %.2f graphs/sec. %.2f%% of %s bytecodes '
          'processed',
          humanize.Commas(sum([len(x) for x in graph_metas_by_output])),
          exported_graph_count / (time.time() - start_time),
          (job_count / len(bytecode_id_chunks)) * 100,
          humanize.DecimalPrefix(len(bytecodes_to_process), ''))

      for output, graph_metas, output in zip(
          self.outputs, graph_metas_by_output, self.outputs):
        if graph_metas:
          with prof.Profile(lambda t: (f"Added {added_to_database} "
                                       f"{output.annotator.name} graph metas")):
            added_to_database = ResilientAddUnique(output.db, graph_metas,
                                                   annotator.name)
            exported_graph_count += added_to_database

    elapsed_time = time.time() - start_time
    app.Log(
        1, 'Exported %s graphs from %s input graphs in %s '
        '(%.2f graphs / second)', humanize.Commas(exported_graph_count),
        humanize.Commas(len(bytecodes_to_process)),
        humanize.Duration(elapsed_time), exported_graph_count / elapsed_time)
    return exported_graph_count


def FetchGraphs(
    annotators: typing.List[GraphAnnotator], bytecode_ids_to_process: np.array,
    input_graph_db_url: str) -> typing.Dict[int, graph_database.GraphMeta]:
  """Read the required graphs and return a map from bytecode ID to graph meta.

  This attempts to read the least possible amount of data from database by only
  reading the graph data strings if they will be later consumed by an annotator.

  Args:
    annotators: The annotators that will consume the bytecode.
    bytecode_ids_to_process: A (annotators,num_bytecodes) shape array of
      bytecode IDs to process.
    input_graph_db_url: The URL of the input graph database.

  Returns:
    A map from bytecode ID to bytecode string.
  """
  load_graphs = any(annotator.requires_graphs for annotator in annotators)

  with prof.Profile(
      lambda t: f"Read {len(bytecode_ids_to_fetch)} input graphs",
      print_to=lambda msg: app.Log(2, msg)):
    # Determine the graph metas that need to be read from the database.
    # Use an ordered list so that we can zip these ids with the return of the
    # query.
    bytecode_ids_to_fetch: typing.List[int] = list(
        sorted(
            set(bytecode_ids_to_process[np.nonzero(bytecode_ids_to_process)])))

    input_db = graph_database.Database(input_graph_db_url)

    # Read the graphs metas from the database. If we need to access the graph
    # data, this is jointly loaded.
    with input_db.Session() as session:
      query = session.query(graph_database.GraphMeta)
      query = query.filter(
          graph_database.GraphMeta.bytecode_id.in_(bytecode_ids_to_fetch))
      # Order by bytecode ID so that we can zip the results with the requested
      # bytecodes.
      query = query.order_by(graph_database.GraphMeta.bytecode_id)

      if load_graphs:
        query = query.options(
            sql.orm.joinedload(graph_database.GraphMeta.graph))

      graph_metas = query.all()
      if len(graph_metas) != len(bytecode_ids_to_fetch):
        raise EnvironmentError(
            "Requested graphs with bytecode IDs "
            f"{bytecode_ids_to_fetch} "
            f"but received {[g.bytecode_id for g in graph_metas]}")

      bytecode_id_to_graph_meta: typing.Dict[int, graph_database.GraphMeta] = {
          id_: row for id_, row in zip(bytecode_ids_to_fetch, graph_metas)
      }
    input_db.Close()  # Don't leave the database connection lying around.
  return bytecode_id_to_graph_meta


def FetchBytecodes(annotators: typing.List[GraphAnnotator],
                   bytecode_ids_to_process: np.array):
  """Read the required bytecoes and return a map from bytecode ID to bytecode
  string.

  This attempts to read the least possible amount of data from database by only
  reading the bytecodes that will later be consumed by an annotator.

  Args:
    annotators: The annotators that will consume the bytecode.
    bytecode_ids_to_process: A (annotators,num_bytecodes) shape array of
      bytecode IDs to process.

  Returns:
    A map from bytecode ID to bytecode string.
  """
  # Optionally read the required bytecodes from the bytecode database. Only read
  # those which are needed by annotators that require bytecodes.
  bytecode_ids_to_fetch: typing.List[int] = []
  for i, ids_for_annotator in enumerate(bytecode_ids_to_process):
    if annotators[i].requires_bytecodes:
      bytecode_ids_to_fetch.extend(
          ids_for_annotator[np.nonzero(ids_for_annotator)])
  # Use an ordered list so that we can zip these ids with the return of the
  # query.
  bytecode_ids_to_fetch = list(sorted(set(bytecode_ids_to_fetch)))

  if bytecode_ids_to_fetch:
    with prof.Profile(
        f"Read {len(bytecode_ids_to_fetch)} bytecode texts",
        print_to=lambda msg: app.Log(2, msg)):
      # Deferred database instantiation so that this script can be run without the
      # --bytecode_db flag set when bytecodes are not required.
      bytecode_db: bytecode_database.Database = FLAGS.bytecode_db()
      with bytecode_db.Session() as session:
        query = session.query(bytecode_database.LlvmBytecode.bytecode)
        query = query.filter(
            bytecode_database.LlvmBytecode.id.in_(bytecode_ids_to_fetch))
        # Order by bytecode ID so that we can zip the results with the requested
        # bytecodes.
        query = query.order_by(bytecode_database.LlvmBytecode.id)

        bytecodes = [row.bytecode for row in query]
      if len(bytecodes) != len(bytecode_ids_to_fetch):
        raise EnvironmentError(f"Requested bytecodes {bytecode_ids_to_fetch} "
                               f"but received {[b.id for b in bytecodes]}")
      bytecode_id_to_bytecode: typing.Dict[int, str] = {
          id_: bytecode
          for id_, bytecode in zip(bytecode_ids_to_fetch, bytecodes)
      }
      bytecode_db.Close()  # Don't leave the database connection lying around.
    return bytecode_id_to_bytecode
  else:
    return {}


def _Worker(packed_args):
  """A graph processor worker. If --nproc > 1, this is called in a worker
  process, hence the packed arguments.
  """
  input_graph_db_url, annotators, bytecode_ids_to_process = packed_args

  bytecode_id_to_graph_meta: typing.Dict[int, graph_database.GraphMeta] = (
      FetchGraphs(annotators, bytecode_ids_to_process, input_graph_db_url))

  bytecode_id_to_bytecode: typing.Dict[int, str] = (FetchBytecodes(
      annotators, bytecode_ids_to_process))

  graph_metas_by_output = []
  for annotator, bytecode_ids in zip(annotators, bytecode_ids_to_process):
    # Ignore the bytecodes that do not need processing.
    bytecode_ids = bytecode_ids[np.nonzero(bytecode_ids)]

    graph_metas = [bytecode_id_to_graph_meta[i] for i in bytecode_ids]
    if annotator.requires_bytecodes:
      bytecodes = [bytecode_id_to_bytecode[i] for i in bytecode_ids]
    else:
      bytecodes = [None] * len(bytecode_ids)

    start_time = time.time()
    annotator_graph_metas = CreateAnnotatedGraphs(annotator, graph_metas,
                                                  bytecodes)
    graph_metas_by_output.append(annotator_graph_metas)
    app.Log(1, "Produced %s %s graphs at bytecodes/sec=%.2f",
            len(annotator_graph_metas), annotator.name,
            len(bytecodes) / (time.time() - start_time))

    # Used by prof.Profile() callback:
    generated_graphs_count = sum([len(x) for x in graph_metas_by_output])

  return graph_metas_by_output


def CreateAnnotatedGraphs(annotator: GraphAnnotator,
                          graph_metas: typing.List[graph_database.GraphMeta],
                          bytecodes: typing.Optional[typing.List[str]] = None):
  """Generate annotated graphs using the given annotator."""
  generated_graph_metas = []

  for i, graph_meta in enumerate(graph_metas):
    # Determine the number of instances to produce based on the size of the
    # input graph.
    n = math.ceil(
        min(graph_meta.node_count / 10, FLAGS.max_instances_per_graph))

    try:
      # Build the arguments list for the graph annotator function.
      kwargs = {
          'n': n,
          'false': np.array([1, 0], dtype=np.float32),
          'true': np.array([0, 1], dtype=np.float32),
      }

      if annotator.requires_graphs:
        # TODO(cec): Rename this argument 'graph' and refactor the graph
        # annotators.
        kwargs['g'] = graph_meta.data
      if annotator.requires_bytecodes:
        kwargs['bytecode'] = bytecodes[i]

      # Run the annotator to produce the annotated graphs.
      annotated_graphs = list(annotator.function(**kwargs))

      # Copy over the graph metadata.
      for annotated_graph in annotated_graphs:
        annotated_graph.group = graph_meta.group
        annotated_graph.bytecode_id = graph_meta.bytecode_id
        annotated_graph.source_name = graph_meta.source_name
        annotated_graph.relpath = graph_meta.relpath
        annotated_graph.language = graph_meta.language

      generated_graph_metas += [
          graph_database.GraphMeta.CreateFromNetworkX(annotated_graph)
          for annotated_graph in annotated_graphs
      ]
    except Exception as e:
      # Insert a zero-node graph meta to mark that exporting this graph failed.
      generated_graph_metas.append(
          graph_database.GraphMeta(
              group=graph_meta.group,
              bytecode_id=graph_meta.bytecode_id,
              source_name=graph_meta.source_name,
              relpath=graph_meta.relpath,
              language=graph_meta.language,
              node_count=0,
              edge_count=0,
              node_embeddings_count=0,
              edge_position_max=0,
              loop_connectedness=0,
              undirected_diameter=0))
      _, _, tb = sys.exc_info()
      tb = traceback.extract_tb(tb, 2)
      filename, line_number, function_name, *_ = tb[-1]
      filename = pathlib.Path(filename).name
      app.Error(
          'Failed to annotate graph for bytecode '
          '%d: %s (%s:%s:%s() -> %s)', graph_meta.bytecode_id, e, filename,
          line_number, function_name,
          type(e).__name__)
      if FLAGS.error:
        raise e

  return generated_graph_metas


def main():
  """Main entry point."""
  input_db = FLAGS.input_graphs_db()

  # Parse the --outputs flag.
  outputs = GetOutputsFromStrings(FLAGS.outputs)
  output_dbs = [x[1] for x in outputs]

  DataFlowAnalysisGraphExporter(outputs)(input_db, output_dbs)


if __name__ == '__main__':
  app.Run(main)
