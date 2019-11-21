"""Print a summary table of model results."""
import io

import pandas as pd
import sqlalchemy as sql

from deeplearning.ml4pl.models import log_database
from labm8 import app
from labm8 import pdutil

app.DEFINE_database('log_db',
                    log_database.Database,
                    None,
                    'The input log database.',
                    must_exist=True)
app.DEFINE_string('format', 'txt',
                  'The format to print the result table. One of {txt,csv}')
app.DEFINE_boolean('human_readable', True,
                   'Format the column data in a human-readable format.')
app.DEFINE_list('extra_model_flags', [], 'Additional model flags to print.')
app.DEFINE_list('extra_flags', [], 'Additional flags to print.')
FLAGS = app.FLAGS


def GetLeaderboard(log_db: log_database.Database,
                   human_readable: bool = False) -> pd.DataFrame:
  """Compute a leaderboard."""
  with log_db.Session() as session:
    # Create a table with batch log stats.
    query = session.query(
        log_database.BatchLogMeta.run_id,
        sql.func.sum(
            log_database.BatchLogMeta.elapsed_time_seconds).label('runtime'),
        sql.func.max(log_database.BatchLogMeta.date_added).label('last_log'),
        sql.func.count(log_database.BatchLogMeta.run_id).label("batches"))
    query = query.group_by(log_database.BatchLogMeta.run_id)
    query = query.group_by(log_database.BatchLogMeta.epoch)
    df = pdutil.QueryToDataFrame(session, query)
    df.set_index(['run_id', 'epoch'], inplace=True)

    # TODO: Print all test stats.

    return df


def main():
  """Main entry point."""
  df = GetLeaderboard(FLAGS.log_db(), human_readable=FLAGS.human_readable)
  if FLAGS.format == 'csv':
    buf = io.StringIO()
    df.to_csv(buf)
    print(buf.getvalue())
  elif FLAGS.format == 'txt':
    print(pdutil.FormatDataFrameAsAsciiTable(df))
  else:
    raise app.UsageError(f"Unknown --format='{FLAGS.format}'")


if __name__ == '__main__':
  app.Run(main)
