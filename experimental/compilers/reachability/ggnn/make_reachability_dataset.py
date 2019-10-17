"""This module prepares a dataset for learning reachability analysis from a
database of control flow graph protocol buffers.
"""

import multiprocessing

import collections
import networkx as nx
import numpy as np
import pathlib
import pickle
import random
import sqlalchemy as sql
import tempfile
import typing

from experimental.compilers.reachability import \
  control_and_data_flow_graph as cdfg
from experimental.compilers.reachability import database
from experimental.compilers.reachability import llvm_util
from experimental.compilers.reachability import reachability_pb2
from experimental.compilers.reachability.ggnn import graph_database
from labm8 import app
from labm8 import decorators
from labm8 import fs
from labm8 import humanize
from labm8 import labtypes
from labm8 import pbutil
from labm8 import prof
from labm8 import sqlutil


app.DEFINE_database(
    'bytecode_db',
    database.Database,
    None,
    'URL of database to read bytecodes from.',
    must_exist=True)
app.DEFINE_database(
    'graph_db', graph_database.Database,
    'sqlite:////var/phd/experimental/compilers/reachability/ggnn/graphs.db',
    'URL of the database to write graphs to.')
app.DEFINE_string('reachability_dataset_type', 'poj104_cfg_only',
                  'The type of dataset to export.')
app.DEFINE_integer('reachability_num_steps', 0,
                   'If > 0, the number of steps to resolve reachability for.')
app.DEFINE_integer(
    'max_bytecodes', 0,
    'If > 0, this limits the number of bytecodes exported to this value.')
app.DEFINE_float(
    'train_to_val_ratio', 1 / 3,
    'The ratio between the size of the validation set relative to the size of '
    'the training set.')
app.DEFINE_float(
    'train_to_test_ratio', 1 / 3,
    'The ratio between the size of the test set relative to the size of the '
    'training set.')
app.DEFINE_integer(
    'reachability_dataset_max_instances_per_graph', 3,
    'The maximum number of reachability graph instances to produce from a '
    'single CDFG. For a CDFG with `n` statements, `n` instances can be '
    'produced by changing the root statement for reachability labels.')
app.DEFINE_integer(
    'reachability_dataset_seed', 0xCEC,
    'The random seed value to use when shuffling graph statements when '
    'selecting the root statement.')
app.DEFINE_integer('reachability_dataset_bytecode_batch_size', 512,
                   'The number of bytecodes to process in a batch.')

FLAGS = app.FLAGS

# Constants.
np_zero = np.array([1, 0], dtype=np.float32)
np_one = np.array([0, 1], dtype=np.float32)


@decorators.timeout(seconds=60)
def AnnotatedGraphToDatabase(g: nx.MultiDiGraph) -> graph_database.GraphMeta:
  # Translate arbitrary node labels into a zero-based index list.
  node_to_index = {node: i for i, node in enumerate(g.nodes)}
  edge_type_to_int = {'control': 0, 'data': 1}
  edge_types = set()

  edge_list = []
  for src, dst, data in g.edges(data=True):
    src_idx = node_to_index[src]
    dst_idx = node_to_index[dst]
    edge_type = edge_type_to_int[data['flow']]
    edge_types.add(edge_type)
    edge_list.append([src_idx, edge_type, dst_idx, -1])

  node_list = [None] * g.number_of_nodes()
  for node, data in g.nodes(data=True):
    node_idx = node_to_index[node]
    node_list[node_idx] = data['x']

  label_list = [None] * g.number_of_nodes()
  for node, data in g.nodes(data=True):
    node_idx = node_to_index[node]
    label_list[node_idx] = data['y']

  graph_dict = {
      'edge_list': edge_list,
      'node_features': node_list,
      'targets': label_list,
  }

  return graph_database.GraphMeta(
      group=g.group,
      bytecode_id=g.bytecode_id,
      source_name=g.source_name,
      relpath=g.relpath,
      language=g.language,
      node_count=g.number_of_nodes(),
      edge_count=g.number_of_edges(),
      edge_type_count=len(edge_types),
      node_features_dimensionality=2,
      node_labels_dimensionality=2,
      data_flow_max_steps_required=g.max_steps_required,
      graph=graph_database.Graph(data=pickle.dumps(graph_dict)))


@decorators.timeout(seconds=60)
def SetReachableNodes(g: nx.MultiDiGraph, root_node: str,
                      max_steps: int) -> None:
  """Annotate nodes in the graph with x, y values for reachability.

  Args:
    g: The graph to annotate.
    root_node: The source node for determining reachability.
    max_steps: The maximum number of steps permitted when computing
      reachability.

  Returns:
    The true maximum number of steps required to compute the annotations.
    In the range 0 < n <= max_steps.
  """
  g.nodes[root_node]['x'] = np_one
  steps = 0

  # Breadth-first traversal to mark all the nodes as reachable.
  visited = set()
  q = collections.deque([(root_node, 0)])
  while q:
    next, steps = q.popleft()
    visited.add(next)
    if not max_steps or steps + 1 <= max_steps:
      for neighbor in cdfg.StatementNeighbors(g, next):
        if neighbor not in visited:
          q.append((neighbor, steps + 1))

    # Mark the node as reachable.
    g.nodes[next]['y'] = np_one

  return steps


def MakeReachabilityAnnotatedGraphs(g: nx.MultiDiGraph,
                                    n: typing.Optional[int] = None
                                   ) -> typing.Iterable[nx.MultiDiGraph]:
  nodes = [node for node, _ in cdfg.StatementNodeIterator(g)]
  n = n or len(nodes)

  for node, data in g.nodes(data=True):
    data['x'] = np_zero
    data['y'] = np_zero

  # If we're taking a sample of nodes to produce graphs (i.e. not all of them),
  # process the nodes in a random order.
  if n < len(nodes):
    random.shuffle(nodes)

  for node in nodes[:n]:
    reachable = g.copy()
    reachable.group = g.group
    reachable.bytecode_id = g.bytecode_id
    reachable.source_name = g.source_name
    reachable.relpath = g.relpath
    reachable.language = g.language
    try:
      reachable.max_steps_required = SetReachableNodes(
          reachable, node, FLAGS.reachability_num_steps)
      yield reachable
    except TimeoutError:
      app.Error("Timeout setting reachable nodes for %s", reachable.bytecode_id)
      pass


def ProcessGroupBytecodeIds(
    db: database.Database,
    group: str,
    make_job_cb,
    process_job_cb,
    bytecode_ids: typing.List[int],
    output_db: graph_database.Database,
    pool: typing.Optional[multiprocessing.Pool] = None) -> int:
  pool = pool or multiprocessing.Pool()
  total_count = 0

  with prof.Profile(lambda t: f'Processed {len(bytecode_ids)} bytecodes '
                    f'({len(bytecode_ids) / t:.2f} bytecode / s)'):
    for i, chunk in enumerate(
        labtypes.Chunkify(bytecode_ids,
                          FLAGS.reachability_dataset_bytecode_batch_size)):
      app.Log(1, 'Processing %s-%s of %s bytecodes (%.2f%%)',
              i * FLAGS.reachability_dataset_bytecode_batch_size,
              i * FLAGS.reachability_dataset_bytecode_batch_size + len(chunk),
              humanize.Commas(len(bytecode_ids)),
              ((i * FLAGS.reachability_dataset_bytecode_batch_size) /
               len(bytecode_ids)) * 100)
      # Run the database queries from the master thread.
      with db.Session() as s:
        jobs = [make_job_cb(s, bytecode_id) for bytecode_id in chunk]

      # Process database query results in parallel.
      graphs = []
      for graphs_chunk in pool.imap_unordered(
          process_job_cb,
          jobs,
          chunksize=max(FLAGS.reachability_dataset_bytecode_batch_size // 16,
                        1)):
        graphs += graphs_chunk

      total_count += len(graphs)
      for graph in graphs:
        graph.group = group
      sqlutil.ResilientAddManyAndCommit(output_db, graphs)
      graphs = []

    if graphs:
      total_count += len(graphs)
      sqlutil.ResilientAddManyAndCommit(output_db, graphs)

  return total_count


def BuildAndRunJobsOnBytecodeIds(
    db: database.Database,
    bytecode_ids: typing.List[int],
    outpath: pathlib.Path,
    make_job_cb,
    process_job_cb,
    pool: typing.Optional[multiprocessing.Pool] = None
) -> typing.List[pathlib.Path]:
  pool = pool or multiprocessing.Pool()
  fragment_paths = []
  data = []

  def EmitFragment(start_idx, end_idx) -> str:
    end_idx = i * FLAGS.reachability_dataset_bytecode_batch_size + len(chunk)
    fragment_path = pathlib.Path(str(outpath) + f'.{start_idx}_{end_idx}')
    with prof.Profile(
        f'Wrote {humanize.Commas(len(data))} graphs to {fragment_path}'):
      with open(fragment_path, 'wb') as f:
        pickle.dump(data, f)
    return fragment_path

  with prof.Profile(lambda t: f'Processed {len(bytecode_ids)} bytecodes '
                    f'({len(bytecode_ids) / t:.2f} bytecode / s)'):
    start_idx = 0
    for i, chunk in enumerate(
        labtypes.Chunkify(bytecode_ids,
                          FLAGS.reachability_dataset_bytecode_batch_size)):
      app.Log(1, 'Processing %s-%s of %s bytecodes (%.2f%%)',
              i * FLAGS.reachability_dataset_bytecode_batch_size,
              i * FLAGS.reachability_dataset_bytecode_batch_size + len(chunk),
              humanize.Commas(len(bytecode_ids)),
              ((i * FLAGS.reachability_dataset_bytecode_batch_size) /
               len(bytecode_ids)) * 100)
      # Run the database queries from the master thread.
      with db.Session() as s:
        jobs = [make_job_cb(s, bytecode_id) for bytecode_id in chunk]

      # Process database query results in parallel.
      for dicts in pool.imap_unordered(
          process_job_cb,
          jobs,
          chunksize=max(FLAGS.reachability_dataset_bytecode_batch_size // 16,
                        1)):
        data += dicts

      if len(data) >= FLAGS.reachability_dataset_file_fragment_size:
        end_idx = i * FLAGS.reachability_dataset_bytecode_batch_size + len(
            chunk)
        fragment_paths.append(EmitFragment(start_idx, end_idx))
        start_idx = end_idx
        data = []

    if data:
      end_idx = len(bytecode_ids)
      fragment_paths.append(EmitFragment(start_idx, end_idx))

  return fragment_paths


def GetAllBytecodeIds(
    db: database.Database,
    train_val_test_ratio: np.array,
) -> typing.Tuple[typing.List[int], typing.List[int], typing.List[int]]:
  """Get the bytecode IDs for the entire database."""
  with db.Session() as s:
    num_bytecodes = s.query(sql.func.count(database.LlvmBytecode.id)).one()[0]
    app.Log(1, "%s total bytecodes in database", humanize.Commas(num_bytecodes))
    # Limit the number of bytecodes if requested.
    if FLAGS.max_bytecodes:
      num_bytecodes = min(num_bytecodes, FLAGS.max_bytecodes)

    ratios = np.floor(train_val_test_ratio * num_bytecodes).astype(np.int32)

    total_count = ratios.sum()
    app.Log(1, 'Loading %s bytecode IDs (%s train, %s val, %s test)',
            humanize.Commas(total_count), humanize.Commas(ratios[0]),
            humanize.Commas(ratios[1]), humanize.Commas(ratios[2]))

    q = s.query(database.LlvmBytecode.id) \
      .order_by(db.Random()) \
      .limit(num_bytecodes)
    ids = [r[0] for r in q]

  train = ids[:ratios[0]]
  val = ids[ratios[0]:ratios[0] + ratios[1]]
  test = ids[ratios[0] + ratios[1]:]
  return train, val, test


def GetPoj104BytecodeIds(
    db: database.Database,
    train_val_test_ratio: typing.Tuple[float, float, float]
) -> typing.Tuple[typing.List[int], typing.List[int], typing.List[int]]:
  """Get the bytecode IDs for the POJ-104 app classification experiment."""

  def GetBytecodeIds(db, filter_cb, bytecode_max: int) -> typing.List[int]:
    with db.Session() as s:
      q = s.query(database.LlvmBytecode.id) \
        .filter(filter_cb())
      if bytecode_max:
        q = q.order_by(db.Random()).limit(bytecode_max)
      return [r[0] for r in q]

  train = lambda: database.LlvmBytecode.source_name == 'poj-104:train'
  test = lambda: database.LlvmBytecode.source_name == 'poj-104:test'
  val = lambda: database.LlvmBytecode.source_name == 'poj-104:val'

  return [
      GetBytecodeIds(db, filter_cb, bytecode_max)
      for filter_cb, bytecode_max in [
          (train, int(FLAGS.max_bytecodes * train_val_test_ratio[0])),
          (val, int(FLAGS.max_bytecodes * train_val_test_ratio[1])),
          (test, int(FLAGS.max_bytecodes * train_val_test_ratio[2])),
      ]
  ]


def ExportDataset(db: database.Database, make_job_cb, process_job_cb,
                  train_ids: typing.List[int], val_ids: typing.List[int],
                  test_ids: typing.List[int],
                  output_db: graph_database.Database):
  train_count = ProcessGroupBytecodeIds(db, "train", make_job_cb,
                                        process_job_cb, train_ids, output_db)
  val_count = ProcessGroupBytecodeIds(db, "val", make_job_cb, process_job_cb,
                                      val_ids, output_db)
  test_count = ProcessGroupBytecodeIds(db, "test", make_job_cb, process_job_cb,
                                       test_ids, output_db)
  app.Log(1, 'Exported %s graphs (%s train, %s val, %s test)',
          humanize.Commas(train_count + val_count + test_count),
          humanize.Commas(train_count), humanize.Commas(val_count),
          humanize.Commas(test_count))


def MakeCfgOnlyJob(s: database.Database.SessionType, bytecode_id: id
                  ) -> typing.Tuple[typing.List[str], str, str, str, int]:

  def GetConstantColumn(rows, column_idx, column_name):
    values = {r[column_idx] for r in rows}
    if len(values) != 1:
      raise ValueError(f'Bytecode ID {bytecode_id} should have the same '
                       f'{column_name} value across its {len(rows)} CFGs, '
                       f'found these values: `{values}`')
    return list(values)[0]

  q = s.query(database.ControlFlowGraphProto.proto,
              database.LlvmBytecode.source_name,
              database.LlvmBytecode.relpath,
              database.LlvmBytecode.language) \
    .join(database.LlvmBytecode) \
    .filter(database.ControlFlowGraphProto.bytecode_id == bytecode_id) \
    .filter(database.ControlFlowGraphProto.status == 0).all()
  # A bytecode may have failed to produce any CFGs.
  if not q:
    app.Warning('Bytecode %s has no CFGs', bytecode_id)
    return None, None, None, None, None
  proto_strings = [r[0] for r in q]
  source = GetConstantColumn(q, 1, 'source')
  relpath = GetConstantColumn(q, 2, 'relpath')
  language = GetConstantColumn(q, 3, 'language')
  return proto_strings, source, relpath, language, bytecode_id


def ProcessCfgOnlyJob(
    packed_args: typing.Tuple[typing.List[str], str, str, str, int]
) -> typing.List[graph_database.GraphMeta]:
  """
  Args:
    packed_args: A packed arguments tuple consisting of a list serialized,
     protos, the source name, the relpath of the bytecode, and the bytecode ID.
  Returns:
    A list of reachability-annotated dictionaries.
  """
  proto_strings, source_name, relpath, language, bytecode_id = packed_args
  if not proto_strings:
    return []
  builder = cdfg.ControlAndDataFlowGraphBuilder(
      dataflow='none',
      preprocess_text=False,
      discard_unknown_statements=False,
  )

  try:
    # Create CFGs from the serialized protos.
    cfgs = []
    proto = reachability_pb2.ControlFlowGraph()
    for proto_string in proto_strings:
      proto.Clear()
      pbutil.FromString(proto_string, proto)
      cfgs.append(llvm_util.LlvmControlFlowGraph.FromProto(proto))

    graphs = [builder.BuildFromControlFlowGraph(cfg) for cfg in cfgs]
    # Ignore "empty" graphs.
    graphs = [g for g in graphs if g.number_of_nodes() and g.number_of_edges()]
    annotated_graphs = []
    for graph in graphs:
      graph.source_name = source_name
      graph.relpath = relpath
      graph.bytecode_id = str(bytecode_id)
      graph.language = language
      annotated_graphs += MakeReachabilityAnnotatedGraphs(
          graph, FLAGS.reachability_dataset_max_instances_per_graph)
    return [AnnotatedGraphToDatabase(graph) for graph in annotated_graphs]
  except Exception as e:
    app.Error('Failed to create CDFG for bytecode %d: %s (%s)', bytecode_id, e,
              type(e).__name__)
    return []


def MakeIcdfgJob(s: database.Database.SessionType,
                 bytecode_id: id) -> typing.Tuple[str, str, str, str, int]:
  q = s.query(database.LlvmBytecode.bytecode,
              database.LlvmBytecode.source_name,
              database.LlvmBytecode.relpath,
              database.LlvmBytecode.language) \
    .filter(database.LlvmBytecode.id == bytecode_id).one()
  bytecode, source, relpath, language = q
  return bytecode, source, relpath, language, bytecode_id


def ProcessIcdfgJob(packed_args: typing.Tuple[str, str, str, str, int]
                   ) -> typing.List[graph_database.GraphMeta]:
  """

  Args:
    packed_args: A packed arguments tuple consisting of a list serialized,
     protos, the source name, the relpath of the bytecode, and the bytecode ID.

  Returns:
    A list of reachability-annotated dictionaries.
  """
  bytecode, source_name, relpath, language, bytecode_id = packed_args
  builder = cdfg.ControlAndDataFlowGraphBuilder(
      dataflow='edges_only',
      preprocess_text=True,
      discard_unknown_statements=False,
  )

  try:
    graph = builder.Build(bytecode)
    graph.source_name = source_name
    graph.bytecode_id = str(bytecode_id)
    graph.language = language
    annotated_graphs = MakeReachabilityAnnotatedGraphs(
        graph, FLAGS.reachability_dataset_max_instances_per_graph)
    return [AnnotatedGraphToDatabase(graph) for graph in annotated_graphs]
  except Exception as e:
    app.Error('Failed to create graph for bytecode %d: %s (%s)', bytecode_id, e,
              type(e).__name__)
    return []


def main():
  """Main entry point."""
  if not FLAGS.bytecode_db:
    raise app.UsageError('--db required')
  db = FLAGS.bytecode_db()
  output_db = FLAGS.graph_db()

  # Temporarily redirect logs to a file, which we will later import into the
  # database's meta table.
  with tempfile.TemporaryDirectory() as d:
    app.LogToDirectory(d, 'log')

    # Record the number of instances per graph that we're generating.
    app.Log(1, 'Generating up to %s instances per graph',
            FLAGS.reachability_dataset_max_instances_per_graph)
    with output_db.Session(commit=True) as s:
      s.query(graph_database.Meta).filter(
          graph_database.Meta.key == 'max_instances_per_graph').delete()
      s.add(
          graph_database.Meta(
              key='max_instances_per_graph',
              value=str(FLAGS.reachability_dataset_max_instances_per_graph)))

    app.Log(1, 'Seeding with %s', FLAGS.reachability_dataset_seed)
    random.seed(FLAGS.reachability_dataset_seed)

    train_val_test_ratio = np.array(
        [1, FLAGS.train_to_val_ratio, FLAGS.train_to_test_ratio])
    train_val_test_ratio /= sum(train_val_test_ratio)

    with prof.Profile('Read bytecode IDs from database'):
      if FLAGS.reachability_dataset_type == 'all_cfg_only':
        train, test, val = GetAllBytecodeIds(db, train_val_test_ratio)
        make_job, process_job = MakeCfgOnlyJob, ProcessCfgOnlyJob
      elif FLAGS.reachability_dataset_type == 'poj104_cfg_only':
        train, test, val = GetPoj104BytecodeIds(db, train_val_test_ratio)
        make_job, process_job = MakeCfgOnlyJob, ProcessCfgOnlyJob
      elif FLAGS.reachability_dataset_type == 'poj104':
        train, test, val = GetPoj104BytecodeIds(db, train_val_test_ratio)
        make_job, process_job = MakeIcdfgJob, ProcessIcdfgJob
      else:
        raise app.UsageError('Unknown value for --reachability_dataset_type')

    with prof.Profile('Exported dataset files'):
      ExportDataset(db, make_job, process_job, train, test, val, output_db)

    log = fs.Read(pathlib.Path(d) / 'log.INFO')
    with output_db.Session(commit=True) as s:
      s.query(graph_database.Meta).filter(
          graph_database.Meta.key == 'log').delete()
      s.add(graph_database.Meta(key='log', value=log))


if __name__ == '__main__':
  app.Run(main)
