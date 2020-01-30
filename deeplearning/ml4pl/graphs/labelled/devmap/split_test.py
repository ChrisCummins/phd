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
"""Unit tests for //deeplearning/ml4pl/graphs/labelled/devmap:split."""
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.labelled.devmap import split
from deeplearning.ml4pl.testing import random_graph_tuple_database_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


@test.Fixture(
  scope="function",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_db"),
)
def empty_graph_db(request) -> graph_tuple_database.Database:
  """A test fixture which yields a graph database with random graph tuples."""
  yield from testing_databases.YieldDatabase(
    graph_tuple_database.Database, request.param
  )


@test.Fixture(
  scope="function",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_db"),
)
def populated_graph_db(request) -> graph_tuple_database.Database:
  """A test fixture which yields a graph database with random graph tuples."""
  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, request.param
  ) as db:
    random_graph_tuple_database_generator.PopulateDatabaseWithRandomGraphTuples(
      db, graph_count=100, graph_y_dimensionality=2
    )
    yield db


@test.Parametrize("k", (3, 5))
@decorators.loop_for(seconds=5, min_iteration_count=3)
def test_fuzz(
  populated_graph_db: graph_tuple_database.Database,
  k: int,
  empty_graph_db: graph_tuple_database.Database,
):
  """Opaque fuzzing of the public methods."""
  splitter = split.StratifiedGraphLabelKFold(k)
  splitter.ApplySplit(populated_graph_db)
  split.CopySplits(populated_graph_db, empty_graph_db)


if __name__ == "__main__":
  test.Main()
