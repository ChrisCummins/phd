"""Copy a graph database.

This is a utility script used to copy the contents of graph databases, or export
their contents to individual pickled files.
"""
import pickle
import typing

import sqlalchemy as sql

from deeplearning.ml4pl.graphs import graph_database
from labm8.py import app
from labm8.py import humanize
from labm8.py import prof

FLAGS = app.FLAGS

app.DEFINE_database(
  "input_db",
  graph_database.Database,
  None,
  "The input database.",
  must_exist=True,
)
app.DEFINE_database(
  "output_db", graph_database.Database, None, "The destination database."
)
app.DEFINE_output_path(
  "output_dir",
  None,
  "The directory to write individual files to. No effect if --output_db is "
  "set.",
)
app.DEFINE_integer("max_rows", 0, "The maximum number of rows to copy.")
app.DEFINE_string("group", None, "Only export graphs from this group.")
app.DEFINE_integer(
  "max_node_count",
  0,
  "Copy only graphs with fewer than this many nodes. If 0, "
  "no limit is applied.",
)


def ChunkedGraphDatabaseReader(
  db: graph_database.Database,
  filters: typing.Optional[typing.List[typing.Callable[[], bool]]] = None,
  chunk_size: int = 256,
  limit: typing.Optional[int] = None,
) -> typing.Iterable[graph_database.GraphMeta]:
  # The order_by_random arguments means that we can't use
  # labm8.py.sqlutil.OffsetLimitBatchedQuery() to read results as each query
  # will produce a different random order. Instead, first run a query to read
  # all of the IDs that match the query, then iterate through the list of IDs in
  # batches.
  with db.Session() as s:
    with prof.Profile(
      lambda t: (
        f"Selected {humanize.Commas(len(ids))} " f"graphs from database"
      )
    ):
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

  if FLAGS.output_db:
    output_db = FLAGS.output_db()
  else:
    output_dir = FLAGS.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

  filters = []
  if FLAGS.max_node_count:
    filters.append(
      lambda: graph_database.GraphMeta.node_count <= FLAGS.max_node_count
    )
  if FLAGS.group:
    filters.append(lambda: graph_database.GraphMeta.group == FLAGS.group)

  for chunk in ChunkedGraphDatabaseReader(
    input_db, filters=filters, limit=FLAGS.max_rows
  ):
    if FLAGS.output_db:
      # Copy the results to a database.
      with output_db.Session(commit=True) as session:
        for row in chunk:
          session.merge(row)
    else:
      # Write each row to a pickled file.
      for row in chunk:
        with open(output_dir / f"{row.id}.pickle", "wb") as f:
          pickle.dump(row.data, f)


if __name__ == "__main__":
  app.Run(main)
