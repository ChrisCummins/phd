"""This module prepares datasets for data flow analyses."""
import itertools
import math
import multiprocessing
import pathlib
import pickle
import random
import sys
import time
import traceback
import typing

import networkx as nx
import numpy as np

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import database_exporters
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled.alias_set import alias_set
from deeplearning.ml4pl.graphs.labelled.datadep import data_dependence
from deeplearning.ml4pl.graphs.labelled.domtree import dominator_tree
from deeplearning.ml4pl.graphs.labelled.liveness import liveness
from deeplearning.ml4pl.graphs.labelled.reachability import reachability
from deeplearning.ml4pl.graphs.labelled.subexpressions import subexpressions
from labm8 import app
from labm8 import humanize
from labm8 import prof
from labm8 import sqlutil

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
app.DEFINE_string('order_by', 'unprocessed', 'The order to process in')
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

  # If true a list of graph IDs is passed to the `function` as named argument
  # `graph_ids`.
  requires_graph_ids: bool = False


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


def GetBytecodeIdsToProcess(input_db: bytecode_database.Database,
                            output_dbs: typing.List[graph_database.Database]
                           ) -> typing.Tuple[typing.List[int], int]:
  """Get a list of bytecode IDs to process."""
  # Read all bytecode IDs.
  with prof.Profile(lambda t: (
      f"Read {humanize.Commas(len(all_bytecode_ids))} input "
      "bytecode IDs")):
    with input_db.Session() as session:
      query = session.query(graph_database.GraphMeta.bytecode_id)
      all_bytecode_ids = set([row.bytecode_id for row in query])

  def GetBytecodesToProcessForOutput(
      output_db: graph_database.Database,
      all_bytecode_ids: typing.Set[int],
  ) -> typing.Set[int]:
    with output_db.Session() as session:
      query = session.query(graph_database.GraphMeta.bytecode_id)
      already_done = set([row.bytecode_id for row in query])
    return all_bytecode_ids - already_done

  with prof.Profile(lambda t: (
      "Read the "
      f"{humanize.Commas(len(all_bytecodes_to_process))} bytecode "
      f"IDs to process")):
    all_bytecodes_to_process_by_output = [
        GetBytecodesToProcessForOutput(output_db, all_bytecode_ids)
        for output_db in output_dbs
    ]

    all_bytecodes_to_process: typing.List[str] = []
    for bytecodes_to_process in all_bytecodes_to_process_by_output:
      all_bytecodes_to_process.extend(list(bytecodes_to_process))

  with prof.Profile(lambda t: (
      f"Selected {humanize.Commas(len(bytecodes_to_process))} of "
      f"{humanize.Commas(len(frequency_table))} bytecodes to "
      "process with "
      f"{humanize.Commas(bytecodes_to_process_by_output[np.nonzero(bytecodes_to_process_by_output)].size)} annotations"
  )):
    if FLAGS.order_by == 'random':
      bytecodes_to_process = np.array(
          list(set(all_bytecodes_to_process)), dtype=np.int32)
      frequency_table = bytecodes_to_process  # Used prof.Profile() callback.
      random.shuffle(bytecodes_to_process)
      num_to_select = min(10000, len(frequency_table) // 3)
      bytecodes_to_process = bytecodes_to_process[:num_to_select]
    else:
      # Create a frequency table how for many times each unprocessed bytecode
      # occurs.
      frequency_table = np.vstack(
          np.unique(all_bytecodes_to_process, return_counts=True)).T
      # Sort the frequency table by count so that most frequently unprocessed
      # bytecodes occur *at the end* of the list.
      sorted_frequency_table = frequency_table[frequency_table[:, 1].argsort()]
      num_to_select = min(10000, len(frequency_table) // 3)
      bytecodes_to_process = frequency_table[:num_to_select, 0]

    # TODO(cec): No need to insert the zero elements here. Remove this, and all
    # of the nonzero() calls.
    bytecodes_to_process_by_output = []
    for all_bytecodes_to_process_for_output in all_bytecodes_to_process_by_output:
      bytecodes_to_process_by_output.append([
          x if x in all_bytecodes_to_process_for_output else 0
          for x in bytecodes_to_process
      ])
    bytecodes_to_process_by_output = np.vstack(bytecodes_to_process_by_output)

  return bytecodes_to_process, bytecodes_to_process_by_output


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
    all_exported_graph_count = 0
    exported_graph_count = self._Export(input_db, pool, batch_size)
    while exported_graph_count:
      all_exported_graph_count += exported_graph_count
      exported_graph_count = self._Export(input_db, pool, batch_size)

    elapsed_time = time.time() - start_time
    app.Log(1, 'Exported %s graphs in %s '
            '(%.2f graphs / second)', humanize.Commas(exported_graph_count),
            humanize.Duration(elapsed_time),
            exported_graph_count / elapsed_time)

  def _Export(self, input_db: sqlutil.Database, pool: multiprocessing.Pool,
              batch_size: int) -> int:
    start_time = time.time()
    exported_graph_count = 0

    bytecodes_to_process, bytecodes_to_process_by_output = GetBytecodeIdsToProcess(
        input_db, [output.db for output in self.outputs])
    if not bytecodes_to_process.shape:
      return 0

    # Break the bytecodes to process into chunks.
    bytecode_id_chunks = np.split(
        bytecodes_to_process_by_output,
        list(range(0, len(bytecodes_to_process_by_output[0]), batch_size))[1:],
        axis=1)
    jobs = list((input_db.url, [x.annotator
                                for x in self.outputs], bytecode_ids_chunk)
                for bytecode_ids_chunk in bytecode_id_chunks)
    app.Log(1, "Divided %s bytecodes into %s %s-bytecode jobs",
            humanize.Commas(len(bytecodes_to_process)), len(bytecode_id_chunks),
            batch_size)

    # Split the jobs into smaller chunks so that imap_unordered doesn't outrun
    # the database writer by too much.
    for jobs_chunk in iter(
        lambda: list(itertools.islice(jobs, FLAGS.nproc * 2)), []):
      app.Log(1, 'Starting chunk of %s jobs', FLAGS.nproc * 2)
      if FLAGS.nproc > 1:
        workers = pool.imap_unordered(_GraphWorker, jobs_chunk)
      else:
        workers = (_GraphWorker(job) for job in jobs_chunk)

      job_count = 0
      for graph_metas_by_output in workers:
        job_count += 1
        exported_graph_count += sum([len(x) for x in graph_metas_by_output])
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
            with prof.Profile(
                f"Added {len(graph_metas)} {output.annotator.name} graph metas"
            ):
              sqlutil.ResilientAddManyAndCommit(output.db, graph_metas)

    elapsed_time = time.time() - start_time
    app.Log(
        1, 'Exported %s graphs from %s input graphs in %s '
        '(%.2f graphs / second)', humanize.Commas(exported_graph_count),
        humanize.Commas(len(bytecodes_to_process)),
        humanize.Duration(elapsed_time), exported_graph_count / elapsed_time)
    return exported_graph_count


def _GraphWorker(packed_args):
  """A graph processor worker. If --multiprocess_database_exporters is set,
  this is called in a worker process.
  """
  input_graph_db_url, annotators, bytecode_ids_to_process = packed_args

  def LoadGraph(group: str, pickled_data: bytes):
    graph = pickle.loads(pickled_data)
    graph.group = group
    return graph

  with prof.Profile(lambda t: f"Read {len(bytecode_ids_to_fetch)} input graphs"
                   ):
    input_graph_db = graph_database.Database(input_graph_db_url)
    # Read the required graphs from the database.
    bytecode_ids_to_fetch = list(
        sorted(
            set(bytecode_ids_to_process[np.nonzero(bytecode_ids_to_process)])))
    with input_graph_db.Session() as session:
      query = session.query(graph_database.GraphMeta.group,
                            graph_database.Graph.pickled_data) \
        .join(graph_database.Graph) \
        .filter(graph_database.GraphMeta.bytecode_id.in_(bytecode_ids_to_fetch)) \
        .order_by(graph_database.GraphMeta.bytecode_id)

      bytecode_id_to_graph: typing.Dict[int, nx.MultiDiGraph] = {
          id_: LoadGraph(*row) for id_, row in zip(bytecode_ids_to_fetch, query)
      }
    input_graph_db.Close()  # Don't leave the database connection lying around.

  # Optionally read the required bytecodes from the bytecode database.
  bytecode_ids_to_fetch = []
  for i, bytecode_ids_for_annotator in enumerate(bytecode_ids_to_process):
    if annotators[i].requires_bytecodes:
      bytecode_ids_to_fetch.extend(
          bytecode_ids_to_process[np.nonzero(bytecode_ids_to_process)])
  bytecode_ids_to_fetch = list(sorted(set(bytecode_ids_to_fetch)))

  if bytecode_ids_to_fetch:
    with prof.Profile(f"Read {len(bytecode_ids_to_fetch)} bytecode texts"):
      # Deferred database instantiation so that this script can be run without the
      # --bytecode_db flag set when bytecodes are not required.
      bytecode_db: bytecode_database.Database = FLAGS.bytecode_db()
      with bytecode_db.Session() as session:
        bytecodes = [
          row.bytecode for row in
          session.query(bytecode_database.LlvmBytecode.bytecode) \
            .filter(bytecode_database.LlvmBytecode.id.in_(bytecode_ids_to_fetch)) \
            .order_by(bytecode_database.LlvmBytecode.id) \
          ]
      bytecode_id_to_bytecode: typing.Dict[int, str] = {
          id_: bytecode
          for id_, bytecode in zip(bytecode_ids_to_fetch, bytecodes)
      }
      bytecode_db.Close()  # Don't leave the database connection lying around.
  else:
    bytecode_id_to_bytecode: typing.Dict[int, str] = {}

  graph_metas_by_output = []
  for annotator, bytecode_ids in zip(annotators, bytecode_ids_to_process):
    # Ignore the bytecodes that do not need processing.
    bytecode_ids = bytecode_ids[np.nonzero(bytecode_ids)]

    graphs = [bytecode_id_to_graph[i] for i in bytecode_ids]
    if annotator.requires_bytecodes:
      bytecodes = [bytecode_id_to_bytecode[i] for i in bytecode_ids]
    else:
      bytecodes = None

    graph_metas_by_output.append(
        CreateAnnotatedGraphs(annotator, graphs, bytecodes))

    # Used by prof.Profile() callback:
    generated_graphs_count = sum([len(x) for x in graph_metas_by_output])

  return graph_metas_by_output


def CreateAnnotatedGraphs(annotator: GraphAnnotator,
                          graphs: typing.List[nx.MultiDiGraph],
                          bytecodes: typing.Optional[typing.List[str]] = None):
  """Generate annotated graphs using the given annotator."""
  graph_metas = []

  for i, graph in enumerate(graphs):
    # Determine the number of instances to produce based on the size of the
    # input graph.
    n = math.ceil(
        min(graph.number_of_nodes() / 10, FLAGS.max_instances_per_graph))

    try:
      with prof.Profile(
          lambda t:
          f"Produced {len(annotated_graphs)} {annotator.name} instances from {graph.number_of_nodes()}-node graph for bytecode {graph.bytecode_id}"
      ):
        if annotator.requires_bytecodes:
          args = (graph, bytecodes[i])
        else:
          args = (graph,)

        annotated_graphs = list(
            annotator.function(
                *args,
                n=n,
                false=np.array([1, 0], dtype=np.float32),
                true=np.array([0, 1], dtype=np.float32)))

        # Copy over graph metadata.
        for annotated_graph in annotated_graphs:
          annotated_graph.group = graph.group
          annotated_graph.bytecode_id = graph.bytecode_id
          annotated_graph.source_name = graph.source_name
          annotated_graph.relpath = graph.relpath
          annotated_graph.language = graph.language
        graph_metas += [
            graph_database.GraphMeta.CreateFromNetworkX(annotated_graph)
            for annotated_graph in annotated_graphs
        ]
    except Exception as e:
      # Insert a zero-node graph to mark that exporting this graph failed.
      graph_metas.append(
          graph_database.GraphMeta(
              group=graph.group,
              bytecode_id=graph.bytecode_id,
              source_name=graph.source_name,
              relpath=graph.relpath,
              language=graph.language,
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
          '%d: %s (%s:%s:%s() -> %s)', graph.bytecode_id, e, filename,
          line_number, function_name,
          type(e).__name__)
      if FLAGS.error:
        raise e

  return graph_metas


def main():
  """Main entry point."""
  input_db = FLAGS.input_graphs_db()

  # Parse the --outputs flag.
  outputs = GetOutputsFromStrings(FLAGS.outputs)
  output_dbs = [x[1] for x in outputs]

  DataFlowAnalysisGraphExporter(outputs)(input_db, output_dbs)


if __name__ == '__main__':
  app.Run(main)
