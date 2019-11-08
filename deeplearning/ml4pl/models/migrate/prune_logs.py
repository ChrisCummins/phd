"""Prune log databases."""
from labm8 import app

from deeplearning.ml4pl.models import log_database

app.DEFINE_database('log_db', log_database.Database, None,
                    'The database to prune.')
app.DEFINE_boolean('prune_runs_with_no_checkpoints', False,
                   "Delete logs for run IDs with no model checkpoints.")

FLAGS = app.FLAGS


def PruneRunsWithNoCheckpoints(log_db: log_database.Database):
  """Remove any logs that do not have a model checkpoint.

  This is useful for "spring cleaning" a database which has a bunch of
  test/failed runs, although note that any job that is currently running but
  hasn't yet reached the end of epoch 1 will be tidied up!
  """
  with log_db.Session() as session:
    query = session.query(
        log_database.ModelCheckpointMeta.run_id.distinct().label('run_id'))
    runs_with_checkpoints = {row.run_id for row in query}

    runs_with_no_checkpoints = set(log_db.run_ids) - runs_with_checkpoints
    app.Log(1, 'Pruning %s run IDs: %s', len(runs_with_no_checkpoints),
            runs_with_no_checkpoints)
    for run_id in runs_with_no_checkpoints:
      log_db.DeleteLogsForRunId(run_id)


def main():
  """Main entry point."""
  log_db: log_database.Database = FLAGS.log_db()

  if FLAGS.prune_runs_with_no_checkpoints:
    PruneRunsWithNoCheckpoints(log_db)


if __name__ == '__main__':
  app.Run(main)
