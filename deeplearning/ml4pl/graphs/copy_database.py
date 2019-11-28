"""Copy a graph database."""
import typing

import sqlalchemy as sql

from deeplearning.ml4pl.graphs import graph_database
from labm8.py import app
from labm8.py import humanize
from labm8.py import prof

FLAGS = app.FLAGS

app.DEFINE_database('input_db',
                    graph_database.Database,
                    None,
                    'The input database.',
                    must_exist=True)
app.DEFINE_database('output_db', graph_database.Database, None,
                    'The destination database.')
app.DEFINE_integer('max_rows', 0, 'The maximum number of rows to copy.')
app.DEFINE_string('group', None, 'Only export graphs from this group.')


def ChunkedGraphDatabaseReader(
    db: graph_database.Database,
    filters: typing.Optional[typing.List[typing.Callable[[], bool]]] = None,
    order_by_random: bool = False,
    chunk_size: int = 256,
    limit: typing.Optional[int] = None
) -> typing.Iterable[graph_database.GraphMeta]:
  # The order_by_random arguments means that we can't use
  # labm8.py.sqlutil.OffsetLimitBatchedQuery() to read results as each query
  # will produce a different random order. Instead, first run a query to read
  # all of the IDs that match the query, then iterate through the list of IDs in
  # batches.
  with db.Session() as s:
    with prof.Profile(lambda t: (f"Selected {humanize.Commas(len(ids))} "
                                 f"graphs from database")):
      q = s.query(graph_database.GraphMeta.id)
      for filter_cb in filters:
        q = q.filter(filter_cb())

      q = q.order_by(db.Random())

      if limit:
        q = q.limit(limit)

      ids = [r[0] for r in q]

    if not ids:
      raise ValueError("Database query returned no results")

    # Iterate through the IDs in batches, running a new query for each.
    while ids:
      batch_ids = ids[:chunk_size]
      ids = ids[chunk_size:]

      q = s.query(graph_database.GraphMeta)
      q = q.options(sql.orm.joinedload(graph_database.GraphMeta.graph))
      q = q.filter(graph_database.GraphMeta.id.in_(batch_ids))
      yield q.all()


def main():
  """Main entry point."""
  input_db = FLAGS.input_db()
  output_db = FLAGS.output_db()

  filters = []
  if FLAGS.group:
    filters.append(lambda: graph_database.GraphMeta.group == FLAGS.group)

  for chunk in ChunkedGraphDatabaseReader(input_db,
                                          order_by_random=True,
                                          filters=filters,
                                          limit=FLAGS.max_rows):
    with output_db.Session(commit=True) as session:
      for row in chunk:
        session.merge(row)


if __name__ == '__main__':
  app.Run(main)
