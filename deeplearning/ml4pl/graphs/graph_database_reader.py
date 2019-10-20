"""A module defining graph database readers."""
import sqlalchemy as sql
import typing

from deeplearning.ml4pl.graphs import graph_database
from labm8 import app
from labm8 import sqlutil

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
  """
  filters = filters or []

  with db.Session() as s:
    # Load both the graph data and the graph eagerly.
    q = s.query(graph_database.GraphMeta)

    if eager_graph_loading:
      q = q.options(sql.orm.joinedload(graph_database.GraphMeta.graph))

    for filter_cb in filters:
      q = q.filter(filter_cb())

    if order_by_random:
      q = q.order_by(db.Random())

    count = 0
    for batch in sqlutil.OffsetLimitBatchedQuery(q, batch_size=buffer_size):
      for row in batch.rows:
        yield row
        count += 1
        if limit and count >= limit:
          return
