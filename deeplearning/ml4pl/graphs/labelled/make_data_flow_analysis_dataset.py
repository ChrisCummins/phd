"""This module prepares datasets for data flow analyses."""
import math
import pathlib
import sys
import traceback
import typing

import numpy as np
import sqlalchemy as sql
from labm8 import app

from deeplearning.ml4pl.graphs import database_exporters
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled.datadep import data_dependence
from deeplearning.ml4pl.graphs.labelled.domtree import dominator_tree
from deeplearning.ml4pl.graphs.labelled.liveness import liveness
from deeplearning.ml4pl.graphs.labelled.reachability import reachability

app.DEFINE_database(
    'input_db',
    graph_database.Database,
    None,
    'URL of database to read pickled networkx graphs from.',
    must_exist=True)
app.DEFINE_database('output_db', graph_database.Database,
                    'sqlite:////var/phd/deeplearning/ml4pl/graphs.db',
                    'URL of the database to write annotated graph tuples to.')
app.DEFINE_string(
    'analysis', 'reachability', 'The data flow to use. One of: '
    '{reachability,domintor_tree,data_dependence,liveness}')
app.DEFINE_string('y_dtype', 'one_hot_float32',
                  'The data type to use for annotating X and Y attributes.')
app.DEFINE_integer(
    'max_instances_per_graph', 10,
    'The maximum number of instances to produce from a single input graph. '
    'For a CDFG with `n` statements, `n` instances can be '
    'produced by changing the root statement for analyses.')

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
  if FLAGS.y_dtype == 'one_hot_float32':
    return (np.array([1, 0], dtype=np.float32),
            np.array([0, 1], dtype=np.float32))
  else:
    raise app.UsageError(f"Unknown y_dtype `{FLAGS.y_dtype}`")


def _ProcessInputs(
    session: graph_database.Database.SessionType,
    bytecode_ids: typing.List[int]) -> typing.List[graph_database.GraphMeta]:
  """Process a set of graphs.

  Returns:
    A list of analysis-annotated graphs.
  """
  jobs = session.query(graph_database.GraphMeta) \
    .filter(graph_database.GraphMeta.bytecode_id.in_(bytecode_ids)) \
    .options(sql.orm.joinedload(graph_database.GraphMeta.graph)) \
    .all()
  session.close()

  annotated_graph_generator = GetAnnotatedGraphGenerator()
  false, true = GetFalseTrueType()

  graph_metas = []
  for input_graph_meta in jobs:
    graph = input_graph_meta.data  # Load pickled networkx graph.

    # Determine the number of instances to produce based on the size of the
    # input graph.
    n = math.ceil(min(g.number_of_nodes() / 10, FLAGS.max_instances_per_graph))

    try:
      annotated_graphs = list(
          annotated_graph_generator(graph, n=n, false=false, true=true))

      # Copy over graph metadata.
      for annotated_graph in annotated_graphs:
        annotated_graph.bytecode_id = graph.bytecode_id
        annotated_graph.source_name = graph.source_name
        annotated_graph.relpath = graph.relpath
        annotated_graph.language = graph.language
      graph_metas += [
          graph_database.GraphMeta.CreateFromNetworkX(annotated_graph)
          for annotated_graph in annotated_graphs
      ]
    except Exception as e:
      _, _, tb = sys.exc_info()
      tb = traceback.extract_tb(tb, 2)
      filename, line_number, function_name, *_ = tb[-1]
      filename = pathlib.Path(filename).name
      app.Error('Failed to annotate graph with id '
                '%d: %s (%s:%s:%s() -> %s)', input_graph_meta.id, e, filename,
                line_number, function_name,
                type(e).__name__)

  return graph_metas


class DataFlowAnalysisGraphExporter(
    database_exporters.GraphDatabaseExporterBase):
  """Add data flow analysis annotations."""

  def GetProcessInputs(self):
    return _ProcessInputs


def _DataFlowExport(input_db, output_db):
  exporter = DataFlowAnalysisGraphExporter(input_db, output_db)
  exporter.Export()


def main():
  """Main entry point."""
  if not FLAGS.input_db:
    raise app.UsageError('--db required')

  database_exporters.Run(FLAGS.input_db(), FLAGS.output_db(), _DataFlowExport)


if __name__ == '__main__':
  app.Run(main)
