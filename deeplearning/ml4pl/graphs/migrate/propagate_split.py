"""Copy the "group" column from one database to another."""
import typing

import sqlalchemy as sql

from deeplearning.ml4pl.graphs import graph_database
from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_database('input_db',
                    graph_database.Database,
                    None,
                    'URL of database to modify.',
                    must_exist=True)
app.DEFINE_database('output_db',
                    graph_database.Database,
                    None,
                    'URL of database to modify.',
                    must_exist=True)


def PropagateGroups(input_db: graph_database.Database,
                    output_db: graph_database.Database) -> typing.List[str]:
  """Propagate the `group` column from one database to another."""
  with input_db.Session() as in_session:
    for group in in_session.query(graph_database.GraphMeta.group).distinct():
      app.Log(1, 'Propagating `%s` group', group)
      query = in_session.query(graph_database.GraphMeta.id) \
          .filter(graph_database.GraphMeta.group == group)
      ids_to_set = [row.id for row in query]

      update = sql.update(graph_database.GraphMeta) \
        .where(graph_database.GraphMeta.id.in_(ids_to_set)) \
        .values(group=group)
      output_db.engine.execute(update)


def main():
  """Main entry point."""
  PropagateGroups(FLAGS.input_db(), FLAGS.output_db())
  app.Log(1, "done")


if __name__ == '__main__':
  app.Run(main)
