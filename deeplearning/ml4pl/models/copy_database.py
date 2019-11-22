"""Copy the contents of a log database."""
from deeplearning.ml4pl.models import log_database
from labm8 import app
from labm8 import prof
from labm8 import sqlutil

FLAGS = app.FLAGS

app.DEFINE_database('input_db',
                    log_database.Database,
                    None,
                    'The input database.',
                    must_exist=True)
app.DEFINE_database('output_db', log_database.Database, None,
                    'The destination database.')
app.DEFINE_string('run_id', None, 'If set, specify the run ID to copy.')


def CopyResults(query, dst_db):
  """Copy the results of the query to a different database."""
  with dst_db.Session(commit=True) as session:
    for chunk in sqlutil.OffsetLimitBatchedQuery(query):
      for row in chunk.rows:
        session.merge(row)


def main():
  """Main entry point."""
  input_db = FLAGS.input_db()
  output_db = FLAGS.output_db()
  run_id = FLAGS.run_id

  with prof.Profile("Copied batch logs"):
    with input_db.Session() as in_session:
      query = in_session.query(log_database.BatchLogMeta.id)
      if run_id:
        query = query.filter(log_database.BatchLogMeta.run_id == run_id)
      batch_logs_to_copy = [row.id for row in query.all()]

      batch_log_metas = in_session.query(log_database.BatchLogMeta)
      batch_log_metas = batch_log_metas.filter(
          log_database.BatchLogMeta.id.in_(batch_logs_to_copy))
      CopyResults(batch_log_metas, output_db)

      batch_logs = in_session.query(log_database.BatchLog)
      batch_logs = batch_logs.filter(
          log_database.BatchLog.id.in_(batch_logs_to_copy))
      CopyResults(batch_logs, output_db)

  with prof.Profile("Copied parameters"):
    with input_db.Session() as in_session:
      params = in_session.query(log_database.Parameter)
      if run_id:
        params = params.filter(log_database.Parameter.run_id == run_id)
      CopyResults(params, output_db)

  with prof.Profile("Copied checkpoints"):
    with input_db.Session() as in_session:
      checkpoints_to_copy = in_session.query(
          log_database.ModelCheckpointMeta.id)
      if run_id:
        checkpoints_to_copy = checkpoints_to_copy.filter(
            log_database.ModelCheckpointMeta.run_id == run_id)

      checkpoint_metas = in_session.query(log_database.ModelCheckpointMeta)
      checkpoint_metas = checkpoint_metas.filter(
          log_database.ModelCheckpointMeta.id.in_(checkpoints_to_copy))
      CopyResults(checkpoint_metas, output_db)

      checkpoints = in_session.query(log_database.ModelCheckpoint)
      checkpoints = checkpoints.filter(
          log_database.ModelCheckpoint.id.in_(checkpoints_to_copy))
      CopyResults(checkpoints, output_db)


if __name__ == '__main__':
  app.Run(main)
