"""Visualize graphs in a database."""
import multiprocessing

import sqlalchemy as sql

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs import graph_viz
from labm8 import app
from labm8 import flags
from labm8 import labtypes

FLAGS = flags.FLAGS

app.DEFINE_database('graph_db', graph_database.Database, None,
                    'The database to read graphs from.')
app.DEFINE_list('id', [], 'A list of graph IDs to visualize.')
app.DEFINE_output_path('outpath',
                       '/tmp/phd/deeplearning/ml4pl/graph_db',
                       'The directory to write dot files to.',
                       is_dir=True)


def GetIdsInOrderWhere(session: graph_database.Database.SessionType, where):
  query = session.query(graph_database.GraphMeta.id)
  query = query.filter(where)
  query = query.order_by(graph_database.GraphMeta.id)
  return [row.id for row in query]


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  FLAGS.outpath.mkdir(exist_okay=True, parents=True)

  graph_db = FLAGS.graph_db()
  with graph_db.Session() as session, multiprocessing.Pool() as pool:
    # First get the full list of IDs to process.
    ids_to_process = []
    ids_to_process += GetIdsInOrderWhere(
        session, lambda: graph_database.GraphMeta.id.in_(FLAGS.id))
    ids_to_process += GetIdsInOrderWhere(
        session,
        lambda: graph_database.GraphMeta.bytecode.in_(FLAGS.bytecode_id))

  # Split the IDs list into chunks.
  for id_chunk in labtypes.Chunkify(ids_to_process, 512):
    # Read all of the graphs in the chunk.
    with graph_db.Session() as session:
      query = session.query(graph_database)
      query = query.options(sql.orm.joinedload(graph_database.GraphMeta.graph))
      query = query.filter(graph_database.GraphMeta.id.in_(id_chunk))
      graph_metas = query.all()

    # Reconstruct the networkx graphs in parallel.
    for input_graph, output_graph in pool.imap_unordered(id_chunk,
                                                         chunksize=64):
      # TODO(github.com/ChrisCummins/ProGraML/issues/3): Implement!
      pass

    input_output_graphs = [(0, 1)]
    for input_graph, output_graph in enumerate(input_output_graphs):
      input_graph_dot = FLAGS.outpath / f'graph_{input_graph.id}_x.dot'
      output_graph_dot = FLAGS.outpath / f'graph_{output_graph.id}_y.dot'

      fs.Write(input_graph_dot,
               graph_viz.GraphToDot(input_graph).encode('utf-8'))
      fs.Write(output_graph_dot,
               graph_viz.GraphToDot(output_graph).encode('utf-8'))


if __name__ == '__main__':
  app.run(main)
