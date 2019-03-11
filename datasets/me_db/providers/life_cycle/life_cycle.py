"""Import data from Life Cycle."""
import multiprocessing
import pathlib
import subprocess
import tempfile
import typing
import zipfile

from datasets.me_db import importers
from datasets.me_db import me_pb2
from labm8 import app
from labm8 import bazelutil
from labm8 import pbutil

FLAGS = app.FLAGS

app.DEFINE_string('life_cycle_inbox', None, 'Inbox to process.')


def ProcessCsvFile(path: pathlib.Path) -> me_pb2.SeriesCollection:
  """Process a LifeCycle CSV data export.

  Args:
    path: Path of the CSV file.

  Returns:
    A SeriesCollection message.

  Raises:
    FileNotFoundError: If the requested file is not found.
  """
  if not path.is_file():
    raise FileNotFoundError(str(path))
  try:
    return pbutil.RunProcessMessageInPlace([
        str(
            bazelutil.DataPath(
                'phd/datasets/me_db/providers/life_cycle/lc_export_csv_worker'))
    ], me_pb2.SeriesCollection(source=str(path)))
  except subprocess.CalledProcessError as e:
    raise importers.ImporterError('LifeCycle', path, str(e)) from e


def ProcessInbox(inbox: pathlib.Path) -> me_pb2.SeriesCollection:
  """Process Life Cycle data in an inbox.

  Args:
    inbox: The inbox path.

  Returns:
    A SeriesCollection message.
  """
  # Do nothing is there is no LC_export.zip file.
  if not (inbox / 'life_cycle' / 'LC_export.zip').is_file():
    return me_pb2.SeriesCollection()

  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    temp_csv = pathlib.Path(d) / 'LC_export.csv'
    with zipfile.ZipFile(inbox / 'life_cycle' / 'LC_export.zip') as z:
      with z.open('LC_export.csv') as csv_in:
        with open(temp_csv, 'wb') as f:
          f.write(csv_in.read())

    return ProcessCsvFile(temp_csv)


def ProcessInboxToQueue(inbox: pathlib.Path, queue: multiprocessing.Queue):
  queue.put(ProcessInbox(inbox))


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  print(pathlib.Path(FLAGS.life_cycle_inbox))


if __name__ == '__main__':
  app.RunWithArgs(main)
