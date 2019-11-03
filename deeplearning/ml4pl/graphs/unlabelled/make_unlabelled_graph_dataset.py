"""This module produces datasets of unlabelled graphs."""
import pathlib
import sys
import traceback
import typing

from labm8 import app
from labm8 import prof

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.bytecode import splitters
from deeplearning.ml4pl.graphs import database_exporters
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.unlabelled.cdfg import \
  control_and_data_flow_graph as cdfg

FLAGS = app.FLAGS

app.DEFINE_database('bytecode_db',
                    bytecode_database.Database,
                    None,
                    'URL of database to read bytecodes from.',
                    must_exist=True)
app.DEFINE_database('graph_db', graph_database.Database,
                    'sqlite:////var/phd/deeplearning/ml4pl/graphs.db',
                    'URL of the database to write unlabelled graph tuples to.')


def _ProcessInputs(
    session: bytecode_database.Database.SessionType,
    bytecode_ids: typing.List[int]) -> typing.List[graph_database.GraphMeta]:
  """Process a set of bytecodes.

  Returns:
    A list of analysis-annotated graphs.
  """
  jobs = session.query(bytecode_database.LlvmBytecode.id,
                       bytecode_database.LlvmBytecode.bytecode,
                       bytecode_database.LlvmBytecode.source_name,
                       bytecode_database.LlvmBytecode.relpath,
                       bytecode_database.LlvmBytecode.language) \
    .filter(bytecode_database.LlvmBytecode.id.in_(bytecode_ids)).all()
  session.close()

  builder = cdfg.ControlAndDataFlowGraphBuilder()

  graph_metas = []

  for bytecode_id, bytecode, source_name, relpath, language in jobs:
    # Haskell uses an older version of LLVM which emits incompatible bytecode.
    # When processing Haskell code we must use the older version of opt. Else,
    # the default version is fine.
    opt = 'opt-3.5' if language == 'haskell' else None

    try:
      with prof.Profile(
          lambda t: f"Constructed {graph.number_of_nodes()}-node CDFG"):
        graph = builder.Build(bytecode, opt=opt)
      graph.bytecode_id = bytecode_id
      graph.source_name = source_name
      graph.relpath = relpath
      graph.language = language
      graph_metas.append(
          graph_database.GraphMeta.CreateWithNetworkXGraph(graph))
    except Exception as e:
      _, _, tb = sys.exc_info()
      tb = traceback.extract_tb(tb, 2)
      filename, line_number, function_name, *_ = tb[-1]
      filename = pathlib.Path(filename).name
      app.Error(
          'Failed to annotate bytecode with id '
          '%d: %s (%s:%s:%s() -> %s)', bytecode_id, e, filename, line_number,
          function_name,
          type(e).__name__)
  return graph_metas


class UnlabelledGraphExporter(database_exporters.BytecodeDatabaseExporterBase):
  """Export unlabelled graphs."""

  def GetProcessInputs(self):
    return _ProcessInputs


def _UnlabelledGraphExport(input_db, output_db):
  groups = splitters.GetGroupsFromFlags(input_db)
  exporter = UnlabelledGraphExporter(input_db, output_db)
  exporter.ExportGroups(groups)


def main():
  """Main entry point."""
  database_exporters.Run(FLAGS.bytecode_db(), FLAGS.graph_db(),
                         _UnlabelledGraphExport)


if __name__ == '__main__':
  app.Run(main)
