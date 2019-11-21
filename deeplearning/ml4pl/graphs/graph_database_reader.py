"""A module for reaching graphs from graph databases."""
import enum
import random
import typing

import sqlalchemy as sql

from deeplearning.ml4pl.graphs import graph_database
from labm8 import app
from labm8 import humanize
from labm8 import prof

FLAGS = app.FLAGS


class BufferedGraphReaderOrder(enum.Enum):
  """Determine the order to read graphs from a database."""
  # In order reading always starts at the smallest graph ID and proceeds
  # incrementally through the graph table.
  IN_ORDER = 1
  # Global random order means that the graphs are selected form the entire graph
  # table using a random order.
  GLOBAL_RANDOM = 2
  # Batch random order means that the graph table is read in order, but once
  # each batch is read, the graphs are then shuffled locally. This aims to
  # strike a balance between the randomness of the graph order and the speed
  # at which graphs can be read, as reading from the table in order generally
  # requires fewer disk seeks than the global random option. Note that this ties
  # the randomness of the graphs to the size of the graph buffer. A larger graph
  # buffer will increase the randomness of the graphs. When buffer_size >= the
  # size of the graph table, this is equivalent to GLOBAL_RANDOM. When
  # buffer_size == 1, this is the same as IN_ORDER.
  BATCH_RANDOM = 3


def BufferedGraphReader(
    db: graph_database.Database,
    filters: typing.Optional[typing.List[typing.Callable[[], bool]]] = None,
    order: BufferedGraphReaderOrder = BufferedGraphReaderOrder.IN_ORDER,
    eager_graph_loading: bool = True,
    buffer_size: int = 256,
    limit: typing.Optional[int] = None,
    print_context: typing.Any = None,
) -> typing.Iterable[graph_database.GraphMeta]:
  """An iterator over the graphs in a database.

  Args:
    db: The database to iterate over the graphs of.
    filters: An optional list of callbacks, where each callback returns a
      filter condition on the GraphMeta table.
    order: Determine the order to read graphs. See BufferedGraphReaderOrder.
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

  with prof.Profile(lambda t: (f"Selected {humanize.Commas(len(ids))} graph")):
    with db.Session() as session:
      # Random ordering means that we can't use
      # labm8.sqlutil.OffsetLimitBatchedQuery() to read results as each query
      # will produce a different random order. Instead, first run a query to
      # read all of the IDs that match the query, then iterate through the list
      # of IDs in batches.
      query = session.query(graph_database.GraphMeta.id)

      # Graphs that fail during dataset generation are inserted as zero-node
      # entries. Ignore those.
      query = query.filter(graph_database.GraphMeta.node_count > 1)

      for filter_cb in filters:
        query = query.filter(filter_cb())

      # If we are ordering with global random then we can scan through the
      # graph table using index range checks, so we need the IDs sorted.
      if order == BufferedGraphReaderOrder.GLOBAL_RANDOM:
        query = query.order_by(db.Random())
      else:
        query = query.order_by(graph_database.GraphMeta.id)

      ids = [r[0] for r in query.all()]

  if not ids:
    raise ValueError(f"Query on database `{db.url}` returned no results: `{q}`")

  # When we are limiting the number of rows and not reading the table in
  # order, pick a random starting point in the list of IDs.
  if limit and order != BufferedGraphReaderOrder.IN_ORDER:
    batch_start = random.randint(0, max(len(ids) - limit - 1, 0))
    ids = ids[batch_start:batch_start + limit]
  elif limit:
    # If we are reading the table in order, we must still respect the limit
    # argument.
    ids = ids[:limit]

  while ids:
    # Peel off a batch of IDs to query.
    batch_ids = ids[:buffer_size]
    ids = ids[buffer_size:]

    query = session.query(graph_database.GraphMeta)

    if eager_graph_loading:
      # Combine the graph data and graph meta queries.
      query = query.options(sql.orm.joinedload(graph_database.GraphMeta.graph))

    # If we are reading in global random order then we must perform ID checks
    # for all IDs in the batch. If not then we have ordered the IDs by value
    # so we can use (faster) index range comparisons.
    if order == BufferedGraphReaderOrder.GLOBAL_RANDOM:
      query = query.filter(graph_database.GraphMeta.id.in_(batch_ids))
    else:
      query = query.filter(graph_database.GraphMeta.id >= batch_ids[0],
                           graph_database.GraphMeta.id <= batch_ids[-1])
      # Note that for index range comparisons we must repeat the same
      # filters as when first getting the graphs.
      query = query.filter(graph_database.GraphMeta.node_count > 1)
      for filter in filters:
        query = query.filter(filter())

    graph_metas = query.all()

    if len(graph_metas) != len(batch_ids):
      raise OSError(f"Requested {len(batch_ids)} graphs in a batch but "
                    f"received {len(graph_metas)}")

    # For batch-level random ordering, shuffle the result of the (in-order)
    # graph query.
    if order == BufferedGraphReaderOrder.BATCH_RANDOM:
      random.shuffle(graph_metas)

    yield from graph_metas
