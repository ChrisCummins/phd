"""Prune log databases."""
import sqlalchemy as sql

from deeplearning.ml4pl.models import log_database
from labm8 import app
from labm8 import humanize

app.DEFINE_database('log_db', log_database.Database, None,
                    'The database to prune.')
app.DEFINE_boolean('prune_runs_with_no_checkpoints', False,
                   "Delete logs for run IDs with no model checkpoints.")
app.DEFINE_boolean('prune_orphans', False,
                   "Delete orphaned child table entries.")
app.DEFINE_list('run_id', [], 'A list of run IDs to delte.')

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


def PruneOrphans(log_db: log_database.Database):
  """Prune orphaned child nodes. Orphans are never a good thing."""
  with log_db.Session() as session:
    checkpoint_ids = [
        row.id for row in session.query(log_database.ModelCheckpointMeta.id)
    ]
    orphans = session.query(log_database.ModelCheckpoint)
    orphans = orphans.filter(
        ~log_database.ModelCheckpoint.id.in_(checkpoint_ids))
    orphaned_checkpoints = [row.id for row in orphans]

    app.Log(1, 'Found %s orphaned checkpoints to delete',
            humanize.Commas(len(orphaned_checkpoints)))
    if orphaned_checkpoints:
      delete = sql.delete(log_database.ModelCheckpoint)
      delete = delete.where(
          log_database.ModelCheckpoint.id.in_(orphaned_checkpoints))
      log_db.engine.execute(delete)

  with log_db.Session() as session:
    batch_log_ids = [
        row.id for row in session.query(log_database.BatchLogMeta.id)
    ]
    orphans = session.query(log_database.BatchLog)
    orphans = orphans.filter(~log_database.BatchLog.id.in_(batch_log_ids))
    orphaned_batch_logs = [row.id for row in orphans]

    app.Log(1, 'Found %s orphaned batch_logs to delete',
            humanize.Commas(len(orphaned_batch_logs)))
    if orphaned_batch_logs:
      delete = sql.delete(log_database.BatchLog)
      delete = delete.where(
          log_database.ModelCheckpoint.id.in_(orphaned_batch_logs))
      log_db.engine.execute(delete)


def main():
  """Main entry point."""
  log_db: log_database.Database = FLAGS.log_db()

  if FLAGS.prune_orphans:
    PruneOrphans(log_db)
  if FLAGS.prune_runs_with_no_checkpoints:
    PruneRunsWithNoCheckpoints(log_db)
  for run_id in FLAGS.run_id:
    log_db.DeleteLogsForRunId(run_id)


if __name__ == '__main__':
  app.Run(main)
