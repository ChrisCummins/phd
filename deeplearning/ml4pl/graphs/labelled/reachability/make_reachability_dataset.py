"""This module prepares a dataset for learning reachability analysis from a
database of control flow graph protocol buffers.
"""

import multiprocessing
import time

import networkx as nx
import numpy as np
import pathlib
import pickle
import random
import sqlalchemy as sql
import tempfile
import typing

from deeplearning.ml4pl import ml4pl_pb2
from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled.reachability import reachability
from deeplearning.ml4pl.graphs.unlabelled.cdfg import \
  control_and_data_flow_graph as cdfg
from deeplearning.ml4pl.graphs.unlabelled.cfg import llvm_util
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
    bytecode_database.Database,
    None,
    'URL of database to read bytecodes from.',
    must_exist=True)
app.DEFINE_database('graph_db', graph_database.Database,
                    'sqlite:////var/phd/deeplearning/ml4pl/graphs.db',
                    'URL of the database to write graphs to.')
app.DEFINE_string('group_type', 'all', 'The type of dataset to export.')
app.DEFINE_string('graph_type', 'cfg_from_proto',
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


@decorators.timeout(seconds=60)
def AnnotatedGraphToDatabase(g: nx.MultiDiGraph) -> graph_database.GraphMeta:
  # Translate arbitrary node labels into a zero-based index list.
  node_to_index = {node: i for i, node in enumerate(g.nodes)}
  edge_type_to_int = {'control': 0, 'data': 1, 'call': 2}
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


def MakeReachabilityAnnotatedGraphs(g: nx.MultiDiGraph,
                                    n: typing.Optional[int] = None
                                   ) -> typing.Iterable[nx.MultiDiGraph]:
  nodes = [node for node, _ in cdfg.StatementNodeIterator(g)]
  n = n or len(nodes)

  # If we're taking a sample of nodes to produce graphs (i.e. not all of them),
  # process the nodes in a random order.
  if n < len(nodes):
    random.shuffle(nodes)

  for node in nodes[:n]:
    reachable = g.copy()
    reachable.bytecode_id = g.bytecode_id
    reachable.source_name = g.source_name
    reachable.relpath = g.relpath
    reachable.language = g.language
    try:
      reachable.max_steps_required = reachability.SetReachableNodes(
          reachable,
          node,
          FLAGS.reachability_num_steps,
          false=np.array([1, 0], dtype=np.float32),
          true=np.array([0, 1], dtype=np.float32))
      yield reachable
    except TimeoutError:
      app.Error("Timeout setting reachable nodes for %s", reachable.bytecode_id)
      pass


def GetAllBytecodeGroups(
    db: bytecode_database.Database,
    train_val_test_ratio: np.array,
) -> typing.Tuple[typing.List[int], typing.List[int], typing.List[int]]:
  """Get the bytecode IDs for the entire database."""
  with db.Session() as s:
    num_bytecodes = s.query(sql.func.count(
        bytecode_database.LlvmBytecode.id)).one()[0]
    app.Log(1, "%s total bytecodes in database", humanize.Commas(num_bytecodes))
    # Limit the number of bytecodes if requested.
    if FLAGS.max_bytecodes:
      num_bytecodes = min(num_bytecodes, FLAGS.max_bytecodes)

    ratios = np.floor(train_val_test_ratio * num_bytecodes).astype(np.int32)

    total_count = ratios.sum()
    app.Log(1, 'Loading %s bytecode IDs (%s train, %s val, %s test)',
            humanize.Commas(total_count), humanize.Commas(ratios[0]),
            humanize.Commas(ratios[1]), humanize.Commas(ratios[2]))

    q = s.query(bytecode_database.LlvmBytecode.id) \
      .order_by(db.Random()) \
      .limit(num_bytecodes)
    ids = [r[0] for r in q]

  train = ids[:ratios[0]]
  val = ids[ratios[0]:ratios[0] + ratios[1]]
  test = ids[ratios[0] + ratios[1]:]

  return {
      'train': train,
      'val': val,
      'test': test,
  }


def GetPoj104BytecodeGroups(
    db: bytecode_database.Database,
    train_val_test_ratio: typing.Tuple[float, float, float]
) -> typing.Tuple[typing.List[int], typing.List[int], typing.List[int]]:
  """Get the bytecode IDs for the POJ-104 app classification experiment."""

  def GetBytecodeIds(filter_cb, limit: int) -> typing.List[int]:
    with db.Session() as session:
      q = session.query(bytecode_database.LlvmBytecode.id) \
          .filter(filter_cb()) \
          .order_by(db.Random()) \
          .limit(limit)
      return [r[0] for r in q]

  train = lambda: bytecode_database.LlvmBytecode.source_name == 'poj-104:train'
  test = lambda: bytecode_database.LlvmBytecode.source_name == 'poj-104:test'
  val = lambda: bytecode_database.LlvmBytecode.source_name == 'poj-104:val'

  with db.Session() as session:
    num_bytecodes = session.query(
        sql.func.count(bytecode_database.LlvmBytecode.id)).one()[0]
  if FLAGS.max_bytecodes:
    num_bytecodes = min(num_bytecodes, FLAGS.max_bytecodes)

  ratios = np.floor(train_val_test_ratio * num_bytecodes).astype(np.int32)

  return {
      "train": GetBytecodeIds(train, ratios[0]),
      "val": GetBytecodeIds(val, ratios[1]),
      "test": GetBytecodeIds(test, ratios[2]),
  }


class DatasetExporterBase(object):

  def __init__(self,
               bytecode_db: bytecode_database.Database,
               graph_db: graph_database.Database,
               pool: typing.Optional[multiprocessing.Pool] = None,
               batch_size: typing.Optional[int] = None):
    self.bytecode_db = bytecode_db
    self.graph_db = graph_db
    self.pool = pool or multiprocessing.Pool()
    self.batch_size = batch_size or FLAGS.reachability_dataset_bytecode_batch_size

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
      chunksize = max(self.batch_size // 16, 8)
      job_processor = self.GetProcessJobFunction()
      workers = self.pool.imap_unordered(
          job_processor, jobs, chunksize=chunksize)
      for graphs_chunk in workers:
        graph_metas += graphs_chunk

      exported_count += len(graph_metas)
      # Set the GraphMeta.group column.
      for graph in graph_metas:
        graph.group = group
      sqlutil.ResilientAddManyAndCommit(self.graph_db, graph_metas)

    return exported_count


ControlFlowGraphJob = typing.Tuple[typing.List[str], str, str, str, int]


def _ProcessControlFlowGraphJob(
    job: ControlFlowGraphJob) -> typing.List[graph_database.GraphMeta]:
  """
  Args:
    job: A packed arguments tuple consisting of a list serialized,
     protos, the source name, the relpath of the bytecode, and the bytecode ID.
  Returns:
    A list of reachability-annotated dictionaries.
  """
  proto_strings, source_name, relpath, language, bytecode_id = job

  builder = cdfg.ControlAndDataFlowGraphBuilder(
      dataflow='none',
      preprocess_text=False,
      discard_unknown_statements=False,
  )

  try:
    # Create CFGs from the serialized protos.
    cfgs = []
    proto = ml4pl_pb2.ControlFlowGraph()
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


class ControlFlowGraphProtoExporter(DatasetExporterBase):

  @staticmethod
  def GetConstantColumn(rows, column_idx, column_name):
    values = {r[column_idx] for r in rows}
    if len(values) != 1:
      raise ValueError(f'Bytecode ID {bytecode_id} should have the same '
                       f'{column_name} value across its {len(rows)} CFGs, '
                       f'found these values: `{values}`')
    return list(values)[0]

  def MakeExportJob(self, session: bytecode_database.Database.SessionType,
                    bytecode_id: id) -> typing.Optional[ControlFlowGraphJob]:
    q = session.query(bytecode_database.ControlFlowGraphProto.proto,
                      bytecode_database.LlvmBytecode.source_name,
                      bytecode_database.LlvmBytecode.relpath,
                      bytecode_database.LlvmBytecode.language) \
      .join(bytecode_database.LlvmBytecode) \
      .filter(bytecode_database.ControlFlowGraphProto.bytecode_id == bytecode_id) \
      .filter(bytecode_database.ControlFlowGraphProto.status == 0).all()
    if not q:
      app.Log(2, 'Bytecode %s has no CFGs, ignoring', bytecode_id)
      return None
    proto_strings = [r[0] for r in q]
    source = self.GetConstantColumn(q, 1, 'source')
    relpath = self.GetConstantColumn(q, 2, 'relpath')
    language = self.GetConstantColumn(q, 3, 'language')
    return proto_strings, source, relpath, language, bytecode_id

  def GetProcessJobFunction(
      self) -> typing.Callable[[ControlFlowGraphJob], typing.
                               List[graph_database.GraphMeta]]:
    return _ProcessControlFlowGraphJob


BytecodeJob = typing.Tuple[str, str, str, str, int]


def _ProcessBytecodeJob(
    job: BytecodeJob) -> typing.List[graph_database.GraphMeta]:
  """

  Args:
    packed_args: A packed arguments tuple consisting of a list serialized,
     protos, the source name, the relpath of the bytecode, and the bytecode ID.

  Returns:
    A list of reachability-annotated dictionaries.
  """
  bytecode, source_name, relpath, language, bytecode_id = job
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


class BytecodeExporter(DatasetExporterBase):

  def MakeExportJob(self, session: bytecode_database.Database.SessionType,
                    bytecode_id: id) -> typing.Optional[BytecodeJob]:
    q = session.query(bytecode_database.LlvmBytecode.bytecode,
                bytecode_database.LlvmBytecode.source_name,
                bytecode_database.LlvmBytecode.relpath,
                bytecode_database.LlvmBytecode.language) \
        .filter(bytecode_database.LlvmBytecode.id == bytecode_id).one()
    bytecode, source, relpath, language = q
    return bytecode, source, relpath, language, bytecode_id

  def GetProcessJobFunction(
      self) -> typing.Callable[[ControlFlowGraphJob], typing.
                               List[graph_database.GraphMeta]]:
    return _ProcessBytecodeJob


def main():
  """Main entry point."""
  if not FLAGS.bytecode_db:
    raise app.UsageError('--db required')
  bytecode_db = FLAGS.bytecode_db()
  graph_db = FLAGS.graph_db()

  # Temporarily redirect logs to a file, which we will later import into the
  # database's meta table.
  with tempfile.TemporaryDirectory() as d:
    app.LogToDirectory(d, 'log')

    # Record the number of instances per graph that we're generating.
    app.Log(1, 'Generating up to %s instances per graph',
            FLAGS.reachability_dataset_max_instances_per_graph)
    with graph_db.Session(commit=True) as s:
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

    with prof.Profile('Read bytecode ID groups from database'):
      if FLAGS.group_type == 'all':
        groups = GetAllBytecodeGroups(bytecode_db, train_val_test_ratio)
      elif FLAGS.group_type == 'poj104':
        groups = GetPoj104BytecodeGroups(bytecode_db, train_val_test_ratio)

    if FLAGS.graph_type == 'cfg_from_proto':
      exporter = ControlFlowGraphProtoExporter(bytecode_db, graph_db)
    elif FLAGS.graph_type == 'icdfg_from_bytecode':
      exporter = BytecodeExporter(bytecode_db, graph_db)
    else:
      raise app.UsageError('Unknown value for --graph_type')

    exporter.ExportGroups(groups)

    log = fs.Read(pathlib.Path(d) / 'log.INFO')
    with graph_db.Session(commit=True) as s:
      s.query(graph_database.Meta).filter(
          graph_database.Meta.key == 'log').delete()
      s.add(graph_database.Meta(key='log', value=log))


if __name__ == '__main__':
  app.Run(main)
