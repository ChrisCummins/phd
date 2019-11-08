"""Delete graphs with a single node."""
from labm8 import app

from deeplearning.ml4pl.graphs import graph_database

FLAGS = app.FLAGS

app.DEFINE_database('graph_db',
                    graph_database.Database,
                    None,
                    'URL of database to modify.',
                    must_exist=True)


def DeleteSingleNodeGraphs(graph_db: graph_database.Database) -> None:
  """Propagate the `group` column from one database to another."""
  with graph_db.Session() as session:
    query = session.query(graph_database.GraphMeta.id) \
      .filter(graph_database.GraphMeta.node_count == 1)
    ids_to_delete = [row.id for row in query]

  graph_db.DeleteGraphs(ids_to_delete)


def main():
  """Main entry point."""
  DeleteSingleNodeGraphs(FLAGS.graph_db())
  app.Log(1, "done")


if __name__ == '__main__':
  app.Run(main)
