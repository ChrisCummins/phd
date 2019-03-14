# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""Import data from YNAB."""
import multiprocessing
import pathlib
import subprocess
import typing

from datasets.me_db import importers
from datasets.me_db import me_pb2
from labm8 import app
from labm8 import bazelutil
from labm8 import pbutil

FLAGS = app.FLAGS

app.DEFINE_string('ynab_inbox', None, 'Inbox to process.')


def ProcessBudgetJsonFile(path: pathlib.Path) -> me_pb2.SeriesCollection:
  if not path.is_file():
    raise FileNotFoundError(str(path))
  try:
    return pbutil.RunProcessMessageInPlace([
        str(
            bazelutil.DataPath(
                'phd/datasets/me_db/providers/ynab/json_budget_worker'))
    ], me_pb2.SeriesCollection(source=str(path)))
  except subprocess.CalledProcessError as e:
    raise importers.ImporterError('LifeCycle', path, str(e)) from e


def ProcessInbox(inbox: pathlib.Path) -> me_pb2.SeriesCollection:
  """Process a directory of YNAB data.

  Args:
    inbox: The inbox path.

  Returns:
    A SeriesCollection message.
  """
  if not (inbox / 'ynab').is_dir():
    return me_pb2.SeriesCollection()

  files = subprocess.check_output(
      ['find', '-L', str(inbox / 'ynab'), '-name', 'Budget.yfull'],
      universal_newlines=True).rstrip().split('\n')

  # TODO(cec): There can be multiple directories for a single budget. Do we need
  # to de-duplicate them?
  files = [pathlib.Path(f) for f in files]

  series_collections = []
  if files and files[0]:
    for file in files:
      series_collections.append(ProcessBudgetJsonFile(file))
  return importers.MergeSeriesCollections(series_collections)


def ProcessInboxToQueue(inbox: pathlib.Path, queue: multiprocessing.Queue):
  queue.put(ProcessInbox(inbox))


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))

  print(ProcessInbox(pathlib.Path(FLAGS.ynab_inbox)))


if __name__ == '__main__':
  app.RunWithArgs(main)
