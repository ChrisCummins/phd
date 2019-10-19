"""This module prepares the POJ-104 dataset for algorithm classification."""

import networkx as nx
import pathlib
import random
import tempfile
import typing
import traceback
import sys

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.bytecode import splitters
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled import database_exporters
from deeplearning.ncc.inst2vec import api as inst2vec
from labm8 import app
from labm8 import fs

app.DEFINE_database(
    'bytecode_db',
    bytecode_database.Database,
    None,
    'URL of database to read bytecodes from.',
    must_exist=True)
app.DEFINE_database('graph_db', graph_database.Database,
                    'sqlite:////var/phd/deeplearning/ml4pl/graphs.db',
                    'URL of the database to write graphs to.')
app.DEFINE_input_path('dictionary', None,
                      'The path of the pickled dictionary to use.')

FLAGS = app.FLAGS


def GetGraphLabel(source_name: str) -> int:
  """Get the algorithm class as an int in the range 0 < x <= 104."""
  # POJ-104 dataset is divided into subdirectories, one for every class.
  # Therefore to get the class, get the directory name.
  return int(source_name.split('/')[0])


def AddXfgFeatures(graph: nx.MultiDiGraph, dictionary) -> None:
  """Add edge features (embedding indices) and graph labels (class num)."""
  flow_translation_table = {
      'ctrl': 'control',
      'data': 'data',
      # TODO(cec): Consolidate XFG `path` flow with `call` flow used elsewhere?
      'path': 'path',
      # Map unlabelled return flow to control paths. Workaround for a bug in
      # inst2vec.
      None: 'control',
      'none': 'control',
  }

  stmt_count = 0
  unknown_count = 0
  # Set edge features.
  for _, _, data in graph.edges(data=True):
    stmt_count += 1
    if 'stmt' not in data:
      raise ValueError(f"`stmt` missing from edge with data: {data}")
    if data['flow'] not in flow_translation_table:
      raise ValueError(f"Unknown edge flow type `{data['flow']}`")

    data['flow'] = flow_translation_table.get(data['flow'], 'ctrl')

    if data['stmt'] in dictionary:
      data['x'] = [dictionary[data['stmt']]]
    else:
      unknown_count += 1
      data['x'] = [dictionary["!UNK"]]

  # Set graph label.
  graph.y = [GetGraphLabel(graph.relpath)]
  return stmt_count, unknown_count


def _ProcessBytecodeJob(
    job: typing.Any) -> typing.List[graph_database.GraphMeta]:
  """Process a bytecode into a POJ-104 XFG graph representation.

  Args:
    packed_args: A packed arguments tuple consisting of a bytecode string,
      the source name, the relpath of the bytecode, the bytecode ID, and
      the vocabular dictionary.

  Returns:
    A list of reachability-annotated dictionaries.
  """
  bytecode, source_name, relpath, language, bytecode_id, dictionary = job

  try:
    graph = inst2vec.LlvmBytecodeToContextualFlowGraph(bytecode)
    graph.source_name = source_name
    graph.relpath = relpath
    graph.bytecode_id = str(bytecode_id)
    graph.language = language

    num_stmts, num_unknowns = AddXfgFeatures(graph, dictionary)

    graph_meta = graph_database.GraphMeta.CreateWithGraphDict(
        graph, {'path', 'control', 'data'})

    # Add additional graph metadata and update graph dict.
    graph_dict = graph_meta.pickled_data
    graph_dict['num_stmts'] = num_stmts
    graph_dict['num_unknowns'] = num_unknowns
    graph_meta.graph = graph_database.Graph.CreatePickled(graph_dict)

    return [graph_meta]
  except Exception as e:
    _, _, tb = sys.exc_info()
    tb = traceback.extract_tb(tb, 2)
    filename, line_number, function_name, *_ = tb[-1]
    filename = pathlib.Path(filename).name
    app.Error('Failed to create graph for bytecode %d: %s (%s:%s:%s() -> %s)',
              bytecode_id, e, filename, line_number, function_name,
              type(e).__name__)
    return []


class Exporter(database_exporters.BytecodeDatabaseExporterBase):
  """Export from LLVM bytecodes."""

  def __init__(self, bytecode_db, graph_db, dictionary: typing.Dict[str, int]):
    super(Exporter, self).__init__(bytecode_db, graph_db)
    self.dictionary = dictionary

  def MakeExportJob(self, session: bytecode_database.Database.SessionType,
                    bytecode_id: id) -> typing.Optional[typing.Any]:
    q = session.query(bytecode_database.LlvmBytecode.bytecode,
                      bytecode_database.LlvmBytecode.source_name,
                      bytecode_database.LlvmBytecode.relpath,
                      bytecode_database.LlvmBytecode.language) \
      .filter(bytecode_database.LlvmBytecode.id == bytecode_id).one()
    bytecode, source, relpath, language = q
    return bytecode, source, relpath, language, bytecode_id, self.dictionary

  def GetProcessJobFunction(
      self
  ) -> typing.Callable[[typing.Any], typing.List[graph_database.GraphMeta]]:
    return _ProcessBytecodeJob

  def Export(self):
    groups = splitters.GetPoj104BytecodeGroups(self.bytecode_db)
    return self.ExportGroups(groups)


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

    exporter = Exporter(bytecode_db, graph_db)
    exporter.Export()

    log = fs.Read(pathlib.Path(d) / 'log.INFO')
    with graph_db.Session(commit=True) as s:
      s.query(graph_database.Meta).filter(
          graph_database.Meta.key == 'log').delete()
      s.add(graph_database.Meta(key='log', value=log))


if __name__ == '__main__':
  app.Run(main)
