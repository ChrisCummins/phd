"""This module prepares the POJ-104 dataset for algorithm classification."""

import pathlib
import random
import tempfile
import typing

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled import database_exporters
from deeplearning.ml4pl.graphs.unlabelled.cdfg import \
  control_and_data_flow_graph as cdfg
from labm8 import app
from labm8 import fs


app.DEFINE_database('bytecode_db',
                    bytecode_database.Database,
                    None,
                    'URL of database to read bytecodes from.',
                    must_exist=True)
app.DEFINE_database('graph_db', graph_database.Database,
                    'sqlite:////var/phd/deeplearning/ml4pl/graphs.db',
                    'URL of the database to write graphs to.')

FLAGS = app.FLAGS


def GetAlgorithmClass(bytecode: bytecode_database.LlvmBytecode) -> int:
  """Get the algorithm class, in the range 0 < x <= 104."""
  # POJ-104 dataset is divided into subdirectories, one for every class.
  # Therefore to get the class, get the directory name.
  return int(bytecode.source_name.split('/')[0])


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
      dataflow=FLAGS.dataflow,
      discard_unknown_statements=False,
  )

  try:
    graph = builder.Build(bytecode)
    graph.source_name = source_name
    graph.bytecode_id = str(bytecode_id)
    graph.language = language
    annotated_graphs = MakeReachabilityAnnotatedGraphs(
        graph, FLAGS.reachability_dataset_max_instances_per_graph)
    return ToGraphMetas(annotated_graphs)
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
      self
  ) -> typing.Callable[[BytecodeJob], typing.List[graph_database.GraphMeta]]:
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

    app.Log(1, 'Seeding with %s', FLAGS.reachability_dataset_seed)
    random.seed(FLAGS.reachability_dataset_seed)

    groups = GetPoj104BytecodeGroups(bytecode_db, train_val_test_ratio)
    exporter = BytecodeExporter(bytecode_db, graph_db)

    exporter.ExportGroups(groups)

    log = fs.Read(pathlib.Path(d) / 'log.INFO')
    with graph_db.Session(commit=True) as s:
      s.query(graph_database.Meta).filter(
          graph_database.Meta.key == 'log').delete()
      s.add(graph_database.Meta(key='log', value=log))


if __name__ == '__main__':
  app.Run(main)
