"""This module prepares a dataset for learning dominator trees from a
database of control flow graph protocol buffers.
"""

import networkx as nx
import numpy as np
import pathlib
import random
import tempfile
import typing

from deeplearning.ml4pl import ml4pl_pb2
from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.bytecode import splitters
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled import database_exporters
from deeplearning.ml4pl.graphs.labelled.domtree import dominator_tree
from deeplearning.ml4pl.graphs.labelled.graph_dict import graph_dict
from deeplearning.ml4pl.graphs.unlabelled.cdfg import \
  control_and_data_flow_graph as cdfg
from deeplearning.ml4pl.graphs.unlabelled.cfg import llvm_util
from labm8 import app
from labm8 import fs
from labm8 import pbutil

app.DEFINE_database(
    'bytecode_db',
    bytecode_database.Database,
    None,
    'URL of database to read bytecodes from.',
    must_exist=True)
app.DEFINE_database('graph_db', graph_database.Database,
                    'sqlite:////var/phd/deeplearning/ml4pl/graphs.db',
                    'URL of the database to write graphs to.')
app.DEFINE_string('graph_type', 'cfg_from_proto',
                  'The type of dataset to export.')
app.DEFINE_string(
    'dataflow', 'none',
    'The type of dataflow annotations to add to generated graphs.')
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
    'max_instances_per_graph', 3,
    'The maximum number of output instances to produce from a single graph. '
    'For a graph with `n` statements, `n` instances can be produced by '
    'changing the root statement.')
app.DEFINE_integer(
    'seed', 0xCEC,
    'The random seed value to use when shuffling graph statements when '
    'selecting the root statement.')
app.DEFINE_boolean('ignore_graph_creation_errors', True,
                   'Ignore errors in graph creation.')

FLAGS = app.FLAGS


def MakeGraphMetas(
    graph: nx.MultiDiGraph) -> typing.Iterable[graph_dict.GraphDict]:
  annotated_graphs = list(
      dominator_tree.MakeDominatorTreeGraphs(
          graph,
          FLAGS.max_instances_per_graph,
          false=np.array([1, 0], dtype=np.float32),
          true=np.array([0, 1], dtype=np.float32)))

  # Copy over graph metadata.
  for annotated_graph in annotated_graphs:
    annotated_graph.bytecode_id = graph.bytecode_id
    annotated_graph.source_name = graph.source_name
    annotated_graph.relpath = graph.relpath
    annotated_graph.language = graph.language

  if FLAGS.dataflow == 'none':
    edge_types = {'control'}
  else:
    edge_types = {'control', 'data'}

  return [
      graph_database.GraphMeta.CreateWithGraphDict(annotated_graph, edge_types)
      for annotated_graph in annotated_graphs
  ]


ControlFlowGraphJob = typing.Tuple[typing.List[str], str, str, str, int]


def _ProcessControlFlowGraphJob(
    job: ControlFlowGraphJob) -> typing.List[graph_database.GraphMeta]:
  """
  Args:
    job: A packed arguments tuple consisting of a list serialized,
     protos, the source name, the relpath of the bytecode, and the bytecode ID.
  Returns:
    A list of annotated dictionaries.
  """
  proto_strings, source_name, relpath, language, bytecode_id = job

  builder = cdfg.ControlAndDataFlowGraphBuilder(
      dataflow=FLAGS.dataflow,
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
    # Ignore single-node graphs (they have no adjacencies).
    graphs = [g for g in graphs if g.number_of_nodes() and g.number_of_edges()]
    graph_metas = []
    for graph in graphs:
      graph.source_name = source_name
      graph.relpath = relpath
      graph.bytecode_id = str(bytecode_id)
      graph.language = language
      graph_metas += MakeGraphMetas(graph)
    return graph_metas
  except Exception as e:
    app.Error('Failed to create CDFG for bytecode %d: %s (%s)', bytecode_id, e,
              type(e).__name__)
    if not FLAGS.ignore_graph_creation_errors:
      raise e
    return []


class ControlFlowGraphProtoExporter(
    database_exporters.BytecodeDatabaseExporterBase):
  """Export from control flow graph protos."""

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
    A list of annotated dictionaries.
  """
  bytecode, source_name, relpath, language, bytecode_id = job
  builder = cdfg.ControlAndDataFlowGraphBuilder(
      dataflow=FLAGS.dataflow,
      discard_unknown_statements=False,
  )

  try:
    graph = builder.Build(bytecode)
    graph.source_name = source_name
    graph.relpath = relpath
    graph.bytecode_id = str(bytecode_id)
    graph.language = language
    return MakeGraphMetas(graph)
  except Exception as e:
    app.Error('Failed to create graph for bytecode %d: %s (%s)', bytecode_id, e,
              type(e).__name__)
    return []


class BytecodeExporter(database_exporters.BytecodeDatabaseExporterBase):
  """Export from LLVM bytecodes."""

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
            FLAGS.max_instances_per_graph)
    with graph_db.Session(commit=True) as s:
      s.query(graph_database.Meta).filter(
          graph_database.Meta.key == 'max_instances_per_graph').delete()
      s.add(
          graph_database.Meta(
              key='max_instances_per_graph',
              value=str(FLAGS.max_instances_per_graph)))

    app.Log(1, 'Seeding with %s', FLAGS.seed)
    random.seed(FLAGS.seed)

    groups = splitters.GetGroupsFromFlags(bytecode_db)

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
