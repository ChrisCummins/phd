"""Copy the contents of a bytecode database."""
import typing

from labm8 import app
from labm8 import humanize
from labm8 import prof

from deeplearning.ml4pl.bytecode import bytecode_database

FLAGS = app.FLAGS

app.DEFINE_database('input_db', bytecode_database.Database, None,
                    'The input database.')
app.DEFINE_database('output_db', bytecode_database.Database, None,
                    'The destination database.')
app.DEFINE_integer('max_rows', 0, 'The maximum number of rows to copy.')


def ChunkedBytecodeDatabaseReader(
    db: bytecode_database.Database,
    filters: typing.Optional[typing.List[typing.Callable[[], bool]]] = None,
    order_by_random: bool = False,
    chunk_size: int = 256,
    limit: typing.Optional[int] = None
) -> typing.Iterable[bytecode_database.LlvmBytecode]:
  filters = filters or []

  # The order_by_random arguments means that we can't use
  # labm8.sqlutil.OffsetLimitBatchedQuery() to read results as each query will
  # produce a different random order. Instead, first run a query to read all of
  # the IDs that match the query, then iterate through the list of IDs in
  # batches.
  with db.Session() as s:
    with prof.Profile(lambda t: (f"Selected {humanize.Commas(len(ids))} "
                                 f"bytecodes from database")):
      q = s.query(bytecode_database.LlvmBytecode.id)
      for filter_cb in filters:
        q = q.filter(filter_cb())

      if order_by_random:
        q = q.order_by(db.Random())

      q = q.limit(limit)

      ids = [r[0] for r in q]

    if not ids:
      raise ValueError("Database query returned no results")

    # Iterate through the IDs in batches, running a new query for each.
    while ids:
      batch_ids = ids[:chunk_size]
      ids = ids[chunk_size:]

      q = s.query(bytecode_database.LlvmBytecode)
      q = q.filter(bytecode_database.LlvmBytecode.id.in_(batch_ids))
      yield q.all()


def main():
  """Main entry point."""
  input_db = FLAGS.input_db()
  output_db = FLAGS.output_db()

  filters = [lambda: bytecode_database.LlvmBytecode.clang_returncode == 0]

  for chunk in ChunkedBytecodeDatabaseReader(input_db,
                                             filters=filters,
                                             order_by_random=True,
                                             limit=FLAGS.max_rows):
    with output_db.Session(commit=True) as session:
      for row in chunk:
        session.merge(row)


if __name__ == '__main__':
  app.Run(main)
