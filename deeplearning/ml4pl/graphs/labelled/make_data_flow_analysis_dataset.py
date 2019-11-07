"""This module prepares datasets for data flow analyses."""
import math
import pathlib
import sys
import traceback
import typing

import numpy as np
import sqlalchemy as sql
from labm8 import app
from labm8 import prof

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import database_exporters
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled.alias_set import alias_set
from deeplearning.ml4pl.graphs.labelled.datadep import data_dependence
from deeplearning.ml4pl.graphs.labelled.domtree import dominator_tree
from deeplearning.ml4pl.graphs.labelled.liveness import liveness
from deeplearning.ml4pl.graphs.labelled.reachability import reachability
from deeplearning.ml4pl.graphs.labelled.subexpressions import subexpressions

app.DEFINE_database('input_db',
                    graph_database.Database,
                    None,
                    'URL of database to read pickled networkx graphs from.',
                    must_exist=True)
app.DEFINE_database('output_db', graph_database.Database,
                    'sqlite:////var/phd/deeplearning/ml4pl/graphs.db',
                    'URL of the database to write annotated graph tuples to.')
app.DEFINE_database('bytecode_db',
                    bytecode_database.Database,
                    None,
                    'URL of database to read bytecode from. Only required when '
                    'analysis requires bytecode.',
                    must_exist=True)
app.DEFINE_string(
    'analysis', 'reachability', 'The data flow to use. One of: '
    '{reachability,dominator_tree,data_dependence,liveness}')
app.DEFINE_string('y_dtype', 'one_hot_float32',
                  'The data type to use for annotating X and Y attributes.')
app.DEFINE_integer(
    'max_instances_per_graph', 10,
    'The maximum number of instances to produce from a single input graph. '
    'For a CDFG with `n` statements, `n` instances can be '
    'produced by changing the root statement for analyses.')

FLAGS = app.FLAGS


class GraphAnnotator(typing.NamedTuple):
  """A named tuple describing a graph annotator, which is a function that
  accepts graphs as inputs and produces labelled graphs."""
  # The function that produces labelled graphs.
  function: typing.Any
  # If true, a list of bytecodes (one for every graph) is passes as the second
  # argument to `function`.
  requires_bytecode: bool = False


def GetAnnotatedGraphGenerator() -> GraphAnnotator:
  """Return the function that generates annotated data flow analysis graphs."""
  if FLAGS.analysis == 'reachability':
    return GraphAnnotator(function=reachability.MakeReachabilityGraphs)
  elif FLAGS.analysis == 'dominator_tree':
    return GraphAnnotator(function=dominator_tree.MakeDominatorTreeGraphs)
  elif FLAGS.analysis == 'data_dependence':
    return GraphAnnotator(function=data_dependence.MakeDataDependencyGraphs)
  elif FLAGS.analysis == 'liveness':
    return GraphAnnotator(function=liveness.MakeLivenessGraphs)
  elif FLAGS.analysis == 'subexpressions':
    return GraphAnnotator(function=subexpressions.MakeSubexpressionsGraphs)
  elif FLAGS.analysis == 'alias_sets':
    return GraphAnnotator(function=alias_set.MakeAliasSetGraphs,
                          requires_bytecode=True)
  else:
    raise app.UsageError(f"Unknown analysis type `{FLAGS.analysis}`")


def GetFalseTrueType():
  """Return the values that should be used for false/true binary labels."""
  if FLAGS.y_dtype == 'one_hot_float32':
    return (np.array([1, 0],
                     dtype=np.float32), np.array([0, 1], dtype=np.float32))
  else:
    raise app.UsageError(f"Unknown y_dtype `{FLAGS.y_dtype}`")


def _ProcessInputs(
    graph_db: graph_database.Database,
    bytecode_ids: typing.List[int]) -> typing.List[graph_database.GraphMeta]:
  """Process a set of graphs.

  Returns:
    A list of analysis-annotated graphs.
  """
  with graph_db.Session() as session:
    graphs_to_process = session.query(graph_database.GraphMeta) \
      .filter(graph_database.GraphMeta.bytecode_id.in_(bytecode_ids)) \
      .options(sql.orm.joinedload(graph_database.GraphMeta.graph)) \
      .order_by(graph_database.GraphMeta.bytecode_id) \
      .all()
  graph_db.Close()  # Don't leave the database connection lying around.

  graph_annotator = GetAnnotatedGraphGenerator()

  # Optionally produce the list of bytecodes to pass to the dataset annotator.
  if graph_annotator.requires_bytecode:
    # Deferred database instantiation so that this script can be run without the
    # --bytecode_db flag set when bytecodes are not required.
    bytecode_db: bytecode_database.Database = FLAGS.bytecode_db()
    with bytecode_db.Session() as session:
      bytecodes = [
        row.bytecode for row in
        session.query(bytecode_database.LlvmBytecode.bytecode) \
        .filter(bytecode_database.LlvmBytecode.id.in_(bytecode_ids)) \
        .order_by(bytecode_database.LlvmBytecode.id) \
      ]

  session.close()

  false, true = GetFalseTrueType()

  graph_metas = []
  for i in range(len(graphs_to_process)):
    input_graph_meta = graphs_to_process[i]
    graph = input_graph_meta.data  # Load pickled networkx graph.

    # Determine the number of instances to produce based on the size of the
    # input graph.
    n = math.ceil(
        min(input_graph_meta.node_count / 10, FLAGS.max_instances_per_graph))

    try:
      with prof.Profile(
          lambda t:
          f"Produced {len(annotated_graphs)} {FLAGS.analysis} instances from {input_graph_meta.node_count}-node graph"
      ):
        if graph_annotator.requires_bytecode:
          bytecode = bytecodes[i]
          annotated_graphs = list(
              graph_annotator.function(graph,
                                       bytecode,
                                       n=n,
                                       false=false,
                                       true=true))
        else:
          annotated_graphs = list(
              graph_annotator.function(graph, n=n, false=false, true=true))

        # Copy over graph metadata.
        for annotated_graph in annotated_graphs:
          annotated_graph.group = input_graph_meta.group
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


def main():
  """Main entry point."""
  DataFlowAnalysisGraphExporter()(FLAGS.input_db(), [FLAGS.output_db()])


if __name__ == '__main__':
  app.Run(main)
