"""Import data from HealthKit."""
import pathlib
import subprocess
import tempfile
import typing
import zipfile
from absl import app
from absl import flags
from absl import logging
from phd.lib.labm8 import bazelutil
from phd.lib.labm8 import pbutil

from datasets.me_db import importers
from datasets.me_db import me_pb2


FLAGS = flags.FLAGS

flags.DEFINE_string('healthkit_inbox', None, 'Inbox to process.')


def ProcessXmlFile(path: pathlib.Path) -> me_pb2.SeriesCollection:
  """Process a HealthKit XML data export.

  Args:
    path: Path of the XML file.

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
                'phd/datasets/me_db/health_kit/xml_export_worker'))],
        me_pb2.SeriesCollection(source=str(path)))
  except subprocess.CalledProcessError as e:
    raise importers.ImporterError('HealthKit', path, str(e)) from e


def ProcessDirectory(directory: pathlib.Path) -> typing.Iterator[
  me_pb2.SeriesCollection]:
  """Process a directory of HealthKit data.

  Args:
    directory: The directory containing the HealthKit data.

  Returns:
    A generator for SeriesCollection messages.
  """
  # Do nothing is there is no LC_export.zip file.
  if not (directory / 'export.zip').is_file():
    return

  logging.info('Unpacking %s', directory / 'export.zip')
  with tempfile.TemporaryDirectory(prefix='phd_') as d:
    temp_xml = pathlib.Path(d) / 'export.xml'
    with zipfile.ZipFile(directory / 'export.zip') as z:
      with z.open('apple_health_export/export.xml') as xml_in:
        with open(temp_xml, 'wb') as f:
          f.write(xml_in.read())

    yield ProcessXmlFile(temp_xml)


def CreateTasksFromInbox(inbox: pathlib.Path) -> importers.ImporterTasks:
  """Generate an iterator of import tasks from an "inbox" directory."""
  if (inbox / 'health_kit').exists():
    yield lambda: ProcessDirectory(inbox / 'health_kit')


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  importers.RunTasksAndExit(
      CreateTasksFromInbox(pathlib.Path(FLAGS.healthkit_inbox)))


if __name__ == '__main__':
  app.run(main)
