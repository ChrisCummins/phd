# Copyright 2019-2020 the ProGraML authors.
#
# Contact Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database_exporter."""
import pathlib

from deeplearning.ml4pl.graphs import programl
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.graphs.unlabelled import (
  unlabelled_graph_database_exporter,
)
from deeplearning.ml4pl.testing import (
  random_unlabelled_graph_database_generator,
)
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import app
from labm8.py import progress
from labm8.py import test


FLAGS = app.FLAGS


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_db"),
)
def immutable_empty_db(request) -> unlabelled_graph_database.Database:
  """A test fixture which yields an empty graph proto database."""
  yield from testing_databases.YieldDatabase(
    unlabelled_graph_database.Database, request.param
  )


IMMUTABLE_TEST_DB_PROTO_COUNT = 100


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_db"),
)
def immutable_test_db(request) -> unlabelled_graph_database.Database:
  with testing_databases.DatabaseContext(
    unlabelled_graph_database.Database, request.param
  ) as db:
    random_unlabelled_graph_database_generator.PopulateDatabaseWithRandomProgramGraphs(
      db, IMMUTABLE_TEST_DB_PROTO_COUNT
    )
    yield db


@test.Parametrize("fmt", list(programl.StdoutGraphFormat))
def test_export_empty_db(
  immutable_empty_db: unlabelled_graph_database.Database,
  tempdir: pathlib.Path,
  fmt: programl.StdoutGraphFormat,
):
  db = immutable_empty_db
  outdir = tempdir / "graphs"
  exporter = unlabelled_graph_database_exporter.GraphDatabaseExporter(
    db=db, outdir=outdir, fmt=fmt,
  )
  progress.Run(exporter)
  assert outdir.is_dir()
  assert len(list(outdir.iterdir())) == 0


@test.Parametrize("fmt", list(programl.StdoutGraphFormat))
def test_export_test_db(
  immutable_test_db: unlabelled_graph_database.Database,
  tempdir: pathlib.Path,
  fmt: programl.StdoutGraphFormat,
):
  db = immutable_test_db
  outdir = tempdir / "graphs"
  exporter = unlabelled_graph_database_exporter.GraphDatabaseExporter(
    db=db, outdir=outdir, fmt=fmt,
  )
  progress.Run(exporter)
  assert outdir.is_dir()
  assert len(list(outdir.iterdir())) == IMMUTABLE_TEST_DB_PROTO_COUNT

  # We can't convert from dot -> graph, so end the test here.
  if fmt == programl.StdoutGraphFormat.DOT:
    return

  # Parse the dumped files.
  for path in outdir.iterdir():
    with open(path, "rb") as f:
      programl.FromBytes(
        f.read(), programl.StdoutGraphFormatToStdinGraphFormat(fmt)
      )


if __name__ == "__main__":
  test.Main()
