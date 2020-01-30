# Copyright (c) 2017-2020 Chris Cummins.
#
# DeepSmith is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# DeepSmith is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with DeepSmith.  If not, see <https://www.gnu.org/licenses/>.
"""A command-line interface for importing protos to the datastore."""
import pathlib
import time

import progressbar

import deeplearning.deepsmith.result
import deeplearning.deepsmith.testcase
from deeplearning.deepsmith import datastore
from deeplearning.deepsmith import db
from labm8.py import app

FLAGS = app.FLAGS

app.DEFINE_list("results", [], "Result proto paths to import")
app.DEFINE_string("results_dir", None, "Directory containing result protos")
app.DEFINE_list("testcases", [], "Testcase proto paths to import")
app.DEFINE_string("testcases_dir", None, "Directory containing testcase protos")
app.DEFINE_boolean(
  "delete_after_import", False, "Delete the proto files after importing."
)


def ImportResultsFromDirectory(
  session: db.session_t, results_dir: pathlib.Path
) -> None:
  """Import Results from a directory of protos.

  Args:
    session: A database session.
    results_dir: Directory containing (only) Result protos.
  """
  files_to_delete = []
  last_commit_time = time.time()
  if not results_dir.is_dir():
    app.Fatal("directory %s does not exist", results_dir)
  for path in progressbar.ProgressBar()(results_dir.iterdir()):
    deeplearning.deepsmith.result.Result.FromFile(session, path)
    files_to_delete.append(path)
    app.Log(1, "Imported result %s", path)
    if time.time() - last_commit_time > 10:
      session.commit()
      if FLAGS.delete_after_import:
        for path in files_to_delete:
          path.unlink()
      files_to_delete = []
      last_commit_time = time.time()
      app.Log(1, "Committed database")
  session.commit()
  if FLAGS.delete_after_import:
    for path in files_to_delete:
      path.unlink()


def ImportTestcasesFromDirectory(
  session: db.session_t, testcases_dir: pathlib.Path
) -> None:
  """Import Testcases from a directory of protos.

  Args:
    session: A database session.
    testcases_dir: Directory containing (only) Testcase protos.
  """
  files_to_delete = []
  last_commit_time = time.time()
  if not testcases_dir.is_dir():
    app.Fatal("directory %s does not exist", testcases_dir)
  for path in progressbar.ProgressBar()(testcases_dir.iterdir()):
    deeplearning.deepsmith.testcase.Testcase.FromFile(session, path)
    files_to_delete.append(path)
    app.Log(1, "Imported testcase %s", path)
    if time.time() - last_commit_time > 10:
      session.commit()
      if FLAGS.delete_after_import:
        for path in files_to_delete:
          path.unlink()
      files_to_delete = []
      last_commit_time = time.time()
      app.Log(1, "Committed database")
  session.commit()
  if FLAGS.delete_after_import:
    for path in files_to_delete:
      path.unlink()


def main(argv):
  del argv
  ds = datastore.DataStore.FromFlags()
  with ds.Session(commit=True) as session:
    last_commit_time = time.time()
    for path in FLAGS.results:
      deeplearning.deepsmith.result.Result.FromFile(session, pathlib.Path(path))
      if time.time() - last_commit_time > 10:
        session.commit()
        last_commit_time = time.time()
    if FLAGS.results_dir:
      ImportResultsFromDirectory(session, pathlib.Path(FLAGS.results_dir))
    for path in FLAGS.testcases:
      deeplearning.deepsmith.testcase.Testcase.FromFile(
        session, pathlib.Path(path)
      )
    session.commit()
    if FLAGS.testcases_dir:
      ImportTestcasesFromDirectory(session, pathlib.Path(FLAGS.testcases_dir))


if __name__ == "__main__":
  app.RunWithArgs(main)
