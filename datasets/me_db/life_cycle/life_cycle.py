"""Import data from Life Cycle."""
import pathlib
import subprocess
import tempfile
import typing
import zipfile

from absl import app
from absl import flags
from phd.lib.labm8 import bazelutil
from phd.lib.labm8 import pbutil

from datasets.me_db import importers
from datasets.me_db import me_pb2


FLAGS = flags.FLAGS

flags.DEFINE_string('life_cycle_inbox', None, 'Inbox to process.')


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
    return pbutil.RunProcessMessageInPlace(
        [str(
            bazelutil.DataPath(
                'phd/datasets/me_db/life_cycle/lc_export_csv_worker'))],
        me_pb2.SeriesCollection(source=str(path)))
  except subprocess.CalledProcessError as e:
    raise importers.ImporterError('LifeCycle', path, str(e)) from e


def ProcessDirectory(directory: pathlib.Path) -> typing.Iterator[
  me_pb2.SeriesCollection]:
  """Process a directory of Life Cycle data.

  Args:
    directory: The directory containing the Life Cycle data.

  Returns:
    A generator for SeriesCollection messages.
  """
  # Do nothing is there is no LC_export.zip file.
  if not (directory / 'LC_export.zip').is_file():
    return

  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    temp_csv = pathlib.Path(d) / 'LC_export.csv'
    with zipfile.ZipFile(directory / 'LC_export.zip') as z:
      with z.open('LC_export.csv') as csv_in:
        with open(temp_csv, 'wb') as f:
          f.write(csv_in.read())

    yield ProcessCsvFile(temp_csv)


def CreateTasksFromInbox(inbox: pathlib.Path) -> importers.ImporterTasks:
  """Generate an iterator of import tasks from an "inbox" directory."""
  if (inbox / 'life_cycle').exists():
    yield lambda: ProcessDirectory(inbox / 'life_cycle')


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  importers.RunTasksAndExit(
      CreateTasksFromInbox(pathlib.Path(FLAGS.life_cycle_inbox)))


if __name__ == '__main__':
  app.run(main)
