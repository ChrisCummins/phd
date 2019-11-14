"""A module defining graph database readers."""
import typing

import sqlalchemy as sql
from labm8 import app
from labm8 import humanize
from labm8 import prof

from deeplearning.ml4pl.graphs import graph_database

FLAGS = app.FLAGS


def BufferedGraphReader(
    db: graph_database.Database,
    filters: typing.Optional[typing.List[typing.Callable[[], bool]]] = None,
    order_by_random: bool = False,
    eager_graph_loading: bool = True,
    buffer_size: int = 256,
    limit: typing.Optional[int] = None
) -> typing.Iterable[graph_database.GraphMeta]:
  """An iterator over the graphs in a database.

  Args:
    db: The database to iterate over the graphs of.
    filters: An optional list of callbacks, where each callback returns a
      filter condition on the graph table.
    order_by_random: If true, return the graphs of the database in a random
      order.
    eager_graph_loading: If true, load the contents of the Graph table eagerly,
      preventing the need for subsequent SQL queries to access the graph data.
    buffer_size: The number of graphs to query from the database at a time. A
      larger number reduces the number of queries, but increases the memory
      requirement.
    limit: Limit the total number of rows returned to this value.

  Raises:
    ValueError: If the query with the given filters returns no results.
  """
  filters = filters or []

  # The order_by_random arguments means that we can't use
  # labm8.sqlutil.OffsetLimitBatchedQuery() to read results as each query will
  # produce a different random order. Instead, first run a query to read all of
  # the IDs that match the query, then iterate through the list of IDs in
  # batches.

  with db.Session() as s:
    with prof.Profile(lambda t: (f"Selected {humanize.Commas(len(ids))} graphs "
                                 "from database"),
                      print_to=lambda msg: app.Log(3, msg)):
      q = s.query(graph_database.GraphMeta.id)
      for filter_cb in filters:
        q = q.filter(filter_cb())

      # Graphs that fail during dataset generation are inserted as zero-node
      # entries. Ignore those.
      q = q.filter(graph_database.GraphMeta.node_count > 1)

      if order_by_random:
        q = q.order_by(db.Random())

      if limit:
        q = q.limit(limit)

      ids = [r[0] for r in q]

    if not ids:
      raise ValueError(
          f"Query on database `{db.url}` returned no results: `{q}`")

    # Iterate through the IDs in batches, running a new query for each.
    while ids:
      batch_ids = ids[:buffer_size]
      ids = ids[buffer_size:]

      q = s.query(graph_database.GraphMeta)
      if eager_graph_loading:
        q = q.options(sql.orm.joinedload(graph_database.GraphMeta.graph))
      q = q.filter(graph_database.GraphMeta.id.in_(batch_ids))

      for graph_meta in q:
        yield graph_meta
