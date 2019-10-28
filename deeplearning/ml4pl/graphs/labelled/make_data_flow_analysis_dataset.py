"""This module prepares datasets for data flow analyses."""
import pathlib
import random
import sys
import tempfile
import traceback
import typing

import networkx as nx
import numpy as np
from labm8 import app
from labm8 import fs
from labm8 import pbutil

from deeplearning.ml4pl import ml4pl_pb2
from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.bytecode import splitters
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled import database_exporters
from deeplearning.ml4pl.graphs.labelled.datadep import data_dependence
from deeplearning.ml4pl.graphs.labelled.domtree import dominator_tree
from deeplearning.ml4pl.graphs.labelled.graph_dict import graph_dict
from deeplearning.ml4pl.graphs.labelled.liveness import liveness
from deeplearning.ml4pl.graphs.labelled.reachability import reachability
from deeplearning.ml4pl.graphs.unlabelled.cdfg import \
  control_and_data_flow_graph as cdfg
from deeplearning.ml4pl.graphs.unlabelled.cfg import llvm_util

app.DEFINE_database('bytecode_db',
                    bytecode_database.Database,
                    None,
                    'URL of database to read bytecodes from.',
                    must_exist=True)
app.DEFINE_database('graph_db', graph_database.Database,
                    'sqlite:////var/phd/deeplearning/ml4pl/graphs.db',
                    'URL of the database to write annotated graphs to.')
app.DEFINE_string(
    'analysis', 'reachability', 'The data flow to use. One of: '
    '{reachability,domintor_tree,data_dependence,liveness}')
app.DEFINE_string('annotation_dtype', 'one_hot_float32',
                  'The data type to use for annotating X and Y attributes.')
app.DEFINE_string('graph_type', 'cfg_from_proto',
                  'The type of dataset to export.')
app.DEFINE_string(
    'dataflow', 'none',
    'The type of data flow annotations to add to generated graphs.')
app.DEFINE_integer(
    'max_instances_per_graph', 3,
    'The maximum number of reachability graph instances to produce from a '
    'single CDFG. For a CDFG with `n` statements, `n` instances can be '
    'produced by changing the root statement for reachability labels.')
app.DEFINE_integer(
    'seed', 0xCEC,
    'The random seed value to use when shuffling graph statements when '
    'selecting the root statement.')

FLAGS = app.FLAGS


def GetAnnotatedGraphGenerator():
  """Return the function that generates annotated data flow analysis graphs."""
  if FLAGS.analysis == 'reachability':
    return reachability.MakeReachabilityGraphs
  elif FLAGS.analysis == 'dominator_tree':
    return dominator_tree.MakeDominatorTreeGraphs
  elif FLAGS.analysis == 'data_dependence':
    return data_dependence.MakeDataDependencyGraphs
  elif FLAGS.analysis == 'liveness':
    return liveness.MakeLivenessGraphs
  else:
    raise app.UsageError(f"Unknown analysis type `{FLAGS.analysis}`")


def GetFalseTrueType():
  """Return the values that should be used for false/true binary labels."""
  if FLAGS.annotation_dtype == 'one_hot_float32':
    return (np.array([1, 0],
                     dtype=np.float32), np.array([0, 1], dtype=np.float32))
  else:
    raise app.UsageError(f"Unknown annotation_dtype `{FLAGS.annotation_dtype}`")


def MakeGraphMetas(graph: nx.MultiDiGraph, annotated_graph_generator, false,
                   true) -> typing.Iterable[graph_dict.GraphDict]:
  """Genereate GraphMeta database rows from the given graph."""
  annotated_graphs = list(
      annotated_graph_generator(graph,
                                FLAGS.max_instances_per_graph,
                                false=false,
                                true=true))

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


def _MakeControlFlowGraphExportJob(
    session: bytecode_database.Database.SessionType,
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
  source = database_exporters.GetConstantColumn(bytecode_id, q, 1, 'source')
  relpath = database_exporters.GetConstantColumn(bytecode_id, q, 2, 'relpath')
  language = database_exporters.GetConstantColumn(bytecode_id, q, 3, 'language')
  return proto_strings, source, relpath, language, bytecode_id


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
      dataflow=FLAGS.dataflow,
      preprocess_text=False,
      discard_unknown_statements=False,
  )

  annotated_graph_generator = GetAnnotatedGraphGenerator()
  false, true = GetFalseTrueType()

  try:
    # Create CFGs from the serialized protos.
    cfgs = []
    proto = ml4pl_pb2.ControlFlowGraph()
    for proto_string in proto_strings:
      proto.Clear()
      pbutil.FromString(proto_string, proto)
      cfgs.append(llvm_util.LlvmControlFlowGraph.FromProto(proto))
  except Exception as e:
    _, _, tb = sys.exc_info()
    tb = traceback.extract_tb(tb, 2)
    filename, line_number, function_name, *_ = tb[-1]
    filename = pathlib.Path(filename).name
    app.Error(
        'Failed to create control flow graphs from bytecode '
        '%d: %s (%s:%s:%s() -> %s)', bytecode_id, e, filename, line_number,
        function_name,
        type(e).__name__)
    return []

  graph_metas = []
  for cfg in cfgs:
    try:
      graph = builder.BuildFromControlFlowGraph(cfg)
      # Ignore single-node graphs (they have no adjacencies).
      if not (graph.number_of_nodes() and graph.number_of_edges()):
        continue

      graph.source_name = source_name
      graph.relpath = relpath
      graph.bytecode_id = str(bytecode_id)
      graph.language = language
      graph_metas += MakeGraphMetas(graph, annotated_graph_generator, false,
                                    true)
    except Exception as e:
      _, _, tb = sys.exc_info()
      tb = traceback.extract_tb(tb, 2)
      filename, line_number, function_name, *_ = tb[-1]
      filename = pathlib.Path(filename).name
      app.Error(
          'Failed to create meta graphs from bytecode '
          '%d: %s (%s:%s:%s() -> %s)', bytecode_id, e, filename, line_number,
          function_name,
          type(e).__name__)

  return graph_metas


class ControlFlowGraphProtoExporter(
    database_exporters.BytecodeDatabaseExporterBase):
  """Export from control flow graph protos."""

  def GetMakeExportJob(self):
    return _MakeControlFlowGraphExportJob

  def GetProcessJobFunction(
      self) -> typing.Callable[[ControlFlowGraphJob], typing.
                               List[graph_database.GraphMeta]]:
    return _ProcessControlFlowGraphJob


BytecodeJob = typing.Tuple[str, str, str, str, int]


def _MakeBytecodeExportJob(session: bytecode_database.Database.SessionType,
                           bytecode_id: id) -> typing.Optional[BytecodeJob]:
  q = session.query(bytecode_database.LlvmBytecode.bytecode,
                    bytecode_database.LlvmBytecode.source_name,
                    bytecode_database.LlvmBytecode.relpath,
                    bytecode_database.LlvmBytecode.language) \
    .filter(bytecode_database.LlvmBytecode.id == bytecode_id).one()
  bytecode, source, relpath, language = q
  return bytecode, source, relpath, language, bytecode_id


def _ProcessBytecodeJob(
    job: BytecodeJob) -> typing.List[graph_database.GraphMeta]:
  """

  Args:
    job: A packed arguments tuple consisting of a list serialized,
     protos, the source name, the relpath of the bytecode, and the bytecode ID.

  Returns:
    A list of reachability-annotated dictionaries.
  """
  bytecode, source_name, relpath, language, bytecode_id = job
  builder = cdfg.ControlAndDataFlowGraphBuilder(
      dataflow=FLAGS.dataflow,
      preprocess_text=False,
      discard_unknown_statements=False,
  )

  annotated_graph_generator = GetAnnotatedGraphGenerator()
  false, true = GetFalseTrueType()

  try:
    graph = builder.Build(bytecode)
    graph.source_name = source_name
    graph.relpath = relpath
    graph.bytecode_id = str(bytecode_id)
    graph.language = language
    return MakeGraphMetas(graph, annotated_graph_generator, false, true)
  except Exception as e:
    app.Error('Failed to create graph for bytecode %d: %s (%s)', bytecode_id, e,
              type(e).__name__)
    return []


class BytecodeExporter(database_exporters.BytecodeDatabaseExporterBase):
  """Export from LLVM bytecodes."""

  def GetMakeExportJob(self):
    return _MakeBytecodeExportJob

  def GetProcessJobFunction(
      self) -> typing.Callable[[ControlFlowGraphJob], typing.
                               List[graph_database.GraphMeta]]:
    return _ProcessBytecodeJob


def Run(exporter_class):
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
          graph_database.Meta(key='max_instances_per_graph',
                              value=str(FLAGS.max_instances_per_graph)))

    app.Log(1, 'Seeding with %s', FLAGS.seed)
    random.seed(FLAGS.seed)

    groups = splitters.GetGroupsFromFlags(bytecode_db)

    exporter = exporter_class(bytecode_db, graph_db)
    exporter.ExportGroups(groups)

    log = fs.Read(pathlib.Path(d) / 'log.INFO')
    with graph_db.Session(commit=True) as s:
      s.query(graph_database.Meta).filter(
          graph_database.Meta.key == 'log').delete()
      s.add(graph_database.Meta(key='log', value=log))


def main():
  """Main entry point."""
  if not FLAGS.bytecode_db:
    raise app.UsageError('--db required')

  if FLAGS.graph_type == 'cfg_from_proto':
    exporter = ControlFlowGraphProtoExporter
  elif FLAGS.graph_type == 'icdfg_from_bytecode':
    exporter = BytecodeExporter
  else:
    raise app.UsageError('Unknown value for --graph_type')

  Run(exporter)


if __name__ == '__main__':
  app.Run(main)
