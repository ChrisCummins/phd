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
import typing

from experimental.compilers.reachability import \
  control_and_data_flow_graph as cdfg
from experimental.compilers.reachability import control_flow_graph
from experimental.compilers.reachability import database
from experimental.compilers.reachability import llvm_util
from experimental.compilers.reachability import reachability_pb2
from labm8 import app
from labm8 import labtypes
from labm8 import pbutil
from labm8 import prof
from labm8 import humanize

app.DEFINE_database(
    'db',
    database.Database,
    None,
    'URL of database to read control flow graphs from.',
    must_exist=True)
app.DEFINE_output_path(
    'outdir',
    '/var/phd/experimental/compilers/reachability/ggnn/dataset',
    'Path to generate GGNN dataset in.',
    is_dir=True)
app.DEFINE_string('reachability_dataset_type', 'poj-104',
                  'The type of dataset to export.')
app.DEFINE_integer(
    'reachability_dataset_max_train_bytecodes', 0,
    'If --reachability_dataset_max_bytecodes > 0, this limits the number of '
    'bytecodes exported to this value.')
app.DEFINE_integer(
    'reachability_dataset_max_val_bytecodes', 0,
    'If --reachability_dataset_max_bytecodes > 0, this limits the number of '
    'bytecodes exported to this value.')
app.DEFINE_integer(
    'reachability_dataset_max_test_bytecodes', 0,
    'If --reachability_dataset_max_bytecodes > 0, this limits the number of '
    'bytecodes exported to this value.')
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


def AnnotatedGraphToDictionary(
    g: nx.MultiDiGraph) -> typing.Dict[str, typing.Any]:
  # Translate arbitrary node labels into a zero-based index list.
  node_to_index = {node: i for i, node in enumerate(g.nodes)}
  edge_type_to_int = {"control": 2, "data": 3}

  edge_list = []
  for src, dst, data in g.edges(data=True):
    src_idx = node_to_index[src]
    dst_idx = node_to_index[dst]
    edge_type = edge_type_to_int[data['flow']]
    edge_list.append([src_idx, edge_type, dst_idx, -1])

  node_list = [None] * g.number_of_nodes()
  for node, data in g.nodes(data=True):
    node_idx = node_to_index[node]
    node_list[node_idx] = data['x']

  label_list = [None] * g.number_of_nodes()
  for node, data in g.nodes(data=True):
    node_idx = node_to_index[node]
    label_list[node_idx] = data['y']

  return {
      'name': g.name,
      'bytecode_id': g.bytecode_id,
      'language': g.language,
      'graph': edge_list,
      'targets': label_list,
      'node_features': node_list,
      'number_of_nodes': g.number_of_nodes(),
  }


def SetReachableNodes(g: nx.MultiDiGraph, root_node: str) -> None:
  g.nodes[root_node]['x'] = np_one

  # Breadth-first traversal to mark all the nodes as reachable.
  visited = set()
  q = collections.deque([root_node])
  while q:
    next = q.popleft()
    visited.add(next)
    for neighbor in cdfg.StatementNeighbors(g, next):
      if neighbor not in visited:
        q.append(neighbor)

    # Mark the node as reachable.
    g.nodes[next]['y'] = np_one


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
    reachable.bytecode_id = g.bytecode_id
    reachable.name = g.name
    reachable.language = g.language
    SetReachableNodes(reachable, node)
    yield reachable


def BytecodeIdToCfgProtos(
    s: database.Database.SessionType,
    bytecode_id: id) -> typing.Tuple[typing.List[str], str, str, str, int]:

  def GetConstantColumn(rows, column_idx, column_name):
    values = {r[column_idx] for r in rows}
    if len(values) != 1:
      raise ValueError(f"Bytecode ID {bytecode_id} should have the same "
                       f"{column_name} value across its {len(rows)} CFGs, "
                       f"found these values: `{values}`")
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
    app.Error("Bytecode %s has no CFGs", bytecode_id)
    return None, None, None, None, None
  proto_strings = [r[0] for r in q]
  source = GetConstantColumn(q, 1, 'source')
  relpath = GetConstantColumn(q, 2, 'relpath')
  language = GetConstantColumn(q, 3, 'language')
  return proto_strings, source, relpath, language, bytecode_id


def CfgProtosToDictionaries(
    packed_args: typing.Tuple[typing.List[str], str, str, str, int]
) -> typing.List[typing.Dict[str, typing.Any]]:
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
      dataflow='edges_only',
      preprocess_text=True,
      discard_unknown_statements=False,
      add_entry_block=False,
      add_exit_block=True,
  )

  try:
    # Create CFGs from the serialized protos.
    cfgs = []
    proto = reachability_pb2.ControlFlowGraph()
    for proto_string in proto_strings:
      proto.Clear()
      pbutil.FromString(proto_string, proto)
      cfgs.append(llvm_util.LlvmControlFlowGraph.FromProto(proto))

    graph = builder.BuildFromControlFlowGraphs(cfgs)
    graph.name = f'{source_name}:{relpath}'
    graph.bytecode_id = str(bytecode_id)
    graph.language = language
    annotated_graphs = MakeReachabilityAnnotatedGraphs(
        graph, FLAGS.reachability_dataset_max_instances_per_graph)
    return [AnnotatedGraphToDictionary(graph) for graph in annotated_graphs]
  except Exception as e:
    app.Error('Failed to create CDFG for bytecode %d: %s (%s)', bytecode_id, e,
              type(e).__name__)
    return []


def ExportBytecodeIdsToFile(db: database.Database,
                            bytecode_ids: typing.List[int],
                            outpath: pathlib.Path,
                            pool: typing.Optional[multiprocessing.Pool] = None):
  pool = pool or multiprocessing.Pool()
  data = []

  with prof.Profile(lambda t: f'Exported {len(bytecode_ids)} bytecodes '
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
        jobs = [BytecodeIdToCfgProtos(s, bytecode_id) for bytecode_id in chunk]

      # Process database query results in parallel.
      for dicts in pool.imap_unordered(
          CfgProtosToDictionaries,
          jobs,
          chunksize=max(FLAGS.reachability_dataset_bytecode_batch_size // 16,
                        1)):
        data += dicts

  with prof.Profile(f'Wrote {humanize.Commas(len(data))} graphs to {outpath}'):
    with open(outpath, 'wb') as f:
      pickle.dump(data, f)


def GetPoj104BytecodeIds(
    db: database.Database
) -> typing.Tuple[typing.List[int], typing.List[int], typing.List[int]]:
  """Get the bytecode IDs for the POJ-104 app classification experiment."""

  def GetBytecodeIds(db, filter_cb, bytecode_max: int) -> typing.List[int]:
    with db.Session() as s:
      q = s.query(database.LlvmBytecode.id) \
        .filter(filter_cb())
      if bytecode_max:
        q = q.limit(bytecode_max)
      return [r[0] for r in q]

  train = lambda: database.LlvmBytecode.source_name == 'poj-104:train'
  test = lambda: database.LlvmBytecode.source_name == 'poj-104:test'
  val = lambda: database.LlvmBytecode.source_name == 'poj-104:val'

  return [
      GetBytecodeIds(db, filter_cb, bytecode_max)
      for filter_cb, bytecode_max in [
          (train, FLAGS.reachability_dataset_max_train_bytecodes),
          (val, FLAGS.reachability_dataset_max_val_bytecodes),
          (test, FLAGS.reachability_dataset_max_test_bytecodes),
      ]
  ]


def ExportDataset(db: database.Database, train_ids: typing.List[int],
                  val_ids: typing.List[int], test_ids: typing.List[int],
                  outdir: pathlib.Path):
  ExportBytecodeIdsToFile(db, train_ids, outdir / 'train.pickle')
  ExportBytecodeIdsToFile(db, val_ids, outdir / 'val.pickle')
  ExportBytecodeIdsToFile(db, test_ids, outdir / 'test.pickle')


def main():
  """Main entry point."""
  if not FLAGS.db:
    raise app.UsageError('--db required')
  db = FLAGS.db()
  outdir = FLAGS.outdir

  app.LogToDirectory('make_reachability_dataset', outdir / 'logs')

  app.Log(1, 'Seeding with %s', FLAGS.reachability_dataset_seed)
  random.seed(FLAGS.reachability_dataset_seed)

  outdir.mkdir(exist_ok=True, parents=True)

  with prof.Profile('Read bytecode IDs from database'):
    if FLAGS.reachability_dataset_type == 'poj-104':
      train, test, val = GetPoj104BytecodeIds(db)
    else:
      raise app.UsageError("Unknown value for --reachability_dataset_type")

  with prof.Profile('Exported dataset files'):
    ExportDataset(db, train, test, val, outdir)


if __name__ == '__main__':
  app.Run(main)
