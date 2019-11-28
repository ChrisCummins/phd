"""Copy run logs from one database to another."""
import sqlalchemy as sql

from deeplearning.ml4pl.models import log_database
from labm8.py import app
from labm8.py import humanize
from labm8.py import sqlutil

app.DEFINE_database('input_db',
                    log_database.Database,
                    None,
                    'The input database.',
                    must_exist=True)
app.DEFINE_database('output_db', log_database.Database, None,
                    'The destination database.')
app.DEFINE_list(
    'run_id', None, 'A list of run IDs to copy. If not provided, all runs are '
    'copied.')

FLAGS = app.FLAGS


def CopyRunId(input_db: log_database.Database, output_db: log_database.Database,
              run_id: str) -> None:
  """Copy the logs for a given run ID. This copies the run parameters, model
  checkpoints, and batch logs.

  Args:
    input_db: The database to copy from.
    output_db: The database to copy to.
    run_id: The name of the run to copy.
  """
  # Copy the parameters.
  with input_db.Session() as in_session, output_db.Session() as out_session:
    query = in_session.query(log_database.Parameter)
    query = query.filter(log_database.Parameter.run_id == run_id)
    if not query.count():
      raise ValueError(f"Run ID {run_id} not found in database")

    for chunk in sqlutil.OffsetLimitBatchedQuery(query,
                                                 batch_size=512,
                                                 compute_max_rows=True):
      app.Log(1, 'Copying %s of %s parameters for run %s',
              humanize.Commas(min(chunk.offset + chunk.limit, chunk.max_rows)),
              humanize.Commas(chunk.max_rows), run_id)
      for row in chunk.rows:
        out_session.merge(row)
      out_session.commit()

  # Copy the model checkpoints.
  with input_db.Session() as in_session, output_db.Session() as out_session:
    query = in_session.query(log_database.ModelCheckpointMeta)
    query = query.filter(log_database.ModelCheckpointMeta.run_id == run_id)
    query = query.options(
        sql.orm.joinedload(log_database.ModelCheckpointMeta.model_checkpoint))

    for chunk in sqlutil.OffsetLimitBatchedQuery(query,
                                                 batch_size=64,
                                                 compute_max_rows=True):
      app.Log(1, 'Copying %s of %s model checkpoints for run %s',
              humanize.Commas(min(chunk.offset + chunk.limit, chunk.max_rows)),
              humanize.Commas(chunk.max_rows), run_id)
      for row in chunk.rows:
        out_session.merge(row)
      out_session.commit()

  # Copy the batch logs.
  with input_db.Session() as in_session, output_db.Session() as out_session:
    query = in_session.query(log_database.BatchLogMeta)
    query = query.filter(log_database.BatchLogMeta.run_id == run_id)
    query = query.options(
        sql.orm.joinedload(log_database.BatchLogMeta.batch_log))

    for chunk in sqlutil.OffsetLimitBatchedQuery(query,
                                                 batch_size=256,
                                                 compute_max_rows=True):
      app.Log(1, 'Copying %s of %s batch logs for run %s',
              humanize.Commas(min(chunk.offset + chunk.limit, chunk.max_rows)),
              humanize.Commas(chunk.max_rows), run_id)
      for row in chunk.rows:
        out_session.merge(row)
      out_session.commit()


def main():
  """Main entry point."""
  input_db: log_database.Database = FLAGS.input_db()
  output_db: log_database.Database = FLAGS.output_db()

  run_ids = FLAGS.run_id or input_db.run_ids

  for run_id in run_ids:
    CopyRunId(input_db, output_db, run_id)


if __name__ == '__main__':
  app.Run(main)
