"""This module prepares the POJ-104 dataset for algorithm classification."""

import networkx as nx
import pathlib
import pickle
import random
import tempfile
import typing

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.bytecode import splitters
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled import database_exporters
from deeplearning.ncc.inst2vec import api as inst2vec
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
  # Set edge features.
  for _, _, data in graph.edges(data=True):
    data['x'] = [dictionary.get(data['stmt'], dictionary["!UNK"])]
    if data['x'][0] == dictionary["!UNK"]:
      print("UNKNOWN", data['stmt'])
    # Map unlabelled return flow to control paths. Workaround for a bug.
    data['flow'] = data.get('flow', 'ctrl')
  # Set graph label.
  graph.y = [GetGraphLabel(graph.relpath)]
  return graph


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
    graph.bytecode_id = str(bytecode_id)
    graph.language = language

    AddXfgFeatures(graph, dictionary)

    graph_dict = graph_database.GraphMeta.CreateWithGraphDict(
        graph, {'path', 'ctrl', 'data'})
    return [graph_dict]
  except Exception as e:
    app.Error('Failed to create graph for bytecode %d: %s (%s)', bytecode_id, e,
              type(e).__name__)
    return []


class Exporter(database_exporters.BytecodeDatabaseExporterBase):
  """Export from LLVM bytecodes."""

  def __init__(self, *args, **kwargs):
    super(Exporter, self).__init__(*args, **kwargs)
    with prof.Profile("Read dictionary from `{FLAGS.dictionary}`"):
      with open(FLAGS.dictionary, 'rb') as f:
        self.dictionary = pickle.load(f)

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
    return self.ExportGroups(self, groups)


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

    exporter = Exporter(bytecode_db, graph_db)
    exporter.Export()

    log = fs.Read(pathlib.Path(d) / 'log.INFO')
    with graph_db.Session(commit=True) as s:
      s.query(graph_database.Meta).filter(
          graph_database.Meta.key == 'log').delete()
      s.add(graph_database.Meta(key='log', value=log))


if __name__ == '__main__':
  app.Run(main)
