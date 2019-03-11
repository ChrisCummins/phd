"""Import data from Timing.app."""
import multiprocessing
import pathlib
import sqlite3
import subprocess
import time
import typing

from datasets.me_db import importers
from datasets.me_db import me_pb2
from labm8 import app
from labm8 import humanize

FLAGS = app.FLAGS

app.DEFINE_string('timing_inbox', None, 'Inbox to process.')


def _ReadDatabaseToSeriesCollection(db) -> me_pb2.SeriesCollection:
  """Extract SeriesCollection from sqlite3 Timing.app database.

  Args:
    db: The sqlite3 database.

  Returns:
    A SeriesCollection message.
  """
  cursor = db.cursor()

  # Construct a map from distinct Task.title columns to Series protos.
  cursor.execute('SELECT DISTINCT(title) FROM TASK')
  title_series_map = {row[0]: me_pb2.Series() for row in cursor.fetchall()}

  # Process data from each title separately.
  for title, series in title_series_map.items():
    start_time = time.time()

    # Set the Series message fields.
    series.family = 'ScreenTime'
    # The name of a series is a CamelCaps version of the Task.title. E.g. 'Web'.
    series.name = "".join(title.title().split())
    series.unit = 'milliseconds'

    # Run a query to aggregate columns data. The SQL engine can do all the heavy
    # lifting, with the only processing of data required being the conversion of
    # Application.title to CamelCaps.
    # TODO(cec): What time zone does Timing.app store results in?
    cursor.execute(
        """
SELECT
  CAST(ROUND(AppActivity.startDate * 1000.0) AS int) as date,
  CAST(ROUND((AppActivity.endDate - AppActivity.startDate) * 1000.0) AS int) as value,
  Application.title as `group`
FROM
  AppActivity
LEFT JOIN 
  Application
  ON AppActivity.applicationID=AppActivity.id
LEFT JOIN
  Task ON AppActivity.taskID=Task.id
WHERE
  Task.title=?
""", (title,))
    # Create Measurement protos for each of the returned rows.
    series.measurement.extend([
        me_pb2.Measurement(
            ms_since_unix_epoch=date,
            value=value,
            group="".join(group.title().split()) if group else "default",
            source='Timing.app',
        ) for date, value, group in cursor
    ])
    app.Log(1, 'Processed %s %s:%s measurements in %.3f seconds',
            humanize.Commas(len(series.measurement)), series.family,
            series.name,
            time.time() - start_time)

  return me_pb2.SeriesCollection(series=title_series_map.values())


def ProcessDatabase(path: pathlib.Path) -> me_pb2.SeriesCollection:
  """Process a Timing.app database.

  Args:
    path: Path of the SQLite database.

  Returns:
    A SeriesCollection message.

  Raises:
    FileNotFoundError: If the requested file is not found.
  """
  if not path.is_file():
    raise FileNotFoundError(str(path))
  try:
    db = sqlite3.connect(str(path))
    return _ReadDatabaseToSeriesCollection(db)
  except subprocess.CalledProcessError as e:
    raise importers.ImporterError('LifeCycle', path, str(e)) from e


def ProcessInbox(inbox: pathlib.Path) -> me_pb2.SeriesCollection:
  """Process Timing.app data in an inbox.

  Args:
    inbox: The inbox path.

  Returns:
    A SeriesCollection message.
  """
  # Do nothing is there is no Timing.app database.
  if not (inbox / 'timing' / 'SQLite.db').is_file():
    return me_pb2.SeriesCollection()

  return ProcessDatabase(inbox / 'timing' / 'SQLite.db')


def ProcessInboxToQueue(inbox: pathlib.Path, queue: multiprocessing.Queue):
  queue.put(ProcessInbox(inbox))


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  print(ProcessInbox(pathlib.Path(FLAGS.timing_inbox)))


if __name__ == '__main__':
  app.RunWithArgs(main)
