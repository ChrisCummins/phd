"""Import data from YNAB."""
import pathlib
import subprocess
import typing
from absl import app
from absl import flags

from datasets.me_db import importers
from datasets.me_db import me_pb2
from lib.labm8 import bazelutil
from lib.labm8 import pbutil


FLAGS = flags.FLAGS

flags.DEFINE_string('ynab_inbox', None, 'Inbox to process.')


def ProcessBudgetJsonFile(path: pathlib.Path) -> me_pb2.SeriesCollection:
  if not path.is_file():
    raise FileNotFoundError(str(path))
  try:
    return pbutil.RunProcessMessageInPlace(
        [str(
            bazelutil.DataPath('phd/datasets/me_db/ynab/json_budget_worker'))],
        me_pb2.SeriesCollection(source=str(path)))
  except subprocess.CalledProcessError as e:
    raise importers.ImporterError('LifeCycle', path, str(e)) from e


def ProcessDirectory(directory: pathlib.Path) -> typing.Iterator[
  me_pb2.SeriesCollection]:
  """Process a directory of YNAB data.

  Args:
    directory: The directory containing the YNAB data.

  Returns:
    A generator for SeriesCollection messages.
  """
  files = subprocess.check_output(
      ['find', '-L', str(directory), '-name', 'Budget.yfull'],
      universal_newlines=True).rstrip().split('\n')

  # TODO(cec): There can be multiple directories for a single budget. Do we need
  # to de-duplicate them?
  files = [pathlib.Path(f) for f in files]

  if files and files[0]:
    print("FILES!!", files)
    for file in files:
      yield ProcessBudgetJsonFile(file)


def CreateTasksFromInbox(inbox: pathlib.Path) -> importers.ImporterTasks:
  """Generate an iterator of import tasks from an "inbox" directory."""
  if (inbox / 'YNAB').exists():
    yield lambda: ProcessDirectory(inbox / 'ynab')


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  importers.RunTasksAndExit(
      CreateTasksFromInbox(pathlib.Path(FLAGS.ynab_inbox)))


if __name__ == '__main__':
  app.run(main)
