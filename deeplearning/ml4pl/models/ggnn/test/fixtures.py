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
"""Fixtures for GGNN tests."""
from typing import List

from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.models import log_database
from deeplearning.ml4pl.models import logger as logging
from deeplearning.ml4pl.testing import random_graph_tuple_database_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import test


FLAGS = test.FLAGS


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("log_db"),
)
def log_db(request) -> log_database.Database:
  """A test fixture which yields an empty log database."""
  yield from testing_databases.YieldDatabase(
    log_database.Database, request.param
  )


@test.Fixture(scope="session")
def logger(log_db: log_database.Database) -> logging.Logger:
  """A test fixture which yields a logger."""
  with logging.Logger(log_db, max_buffer_length=128) as logger:
    yield logger


@test.Fixture(
  scope="session", params=(0, 2), namer=lambda x: "graph_x_dimensionality:{x}"
)
def graph_x_dimensionality(request) -> int:
  """A test fixture which enumerates graph feature dimensionalities."""
  return request.param


@test.Fixture(
  scope="session",
  params=(2, 104),
  namer=lambda x: f"graph_y_dimensionality:{x}",
)
def graph_y_dimensionality(request) -> int:
  """A test fixture which enumerates graph label dimensionalities."""
  return request.param


@test.Fixture(
  scope="session", params=(2, 3), namer=lambda x: f"node_y_dimensionality:{x}"
)
def node_y_dimensionality(request) -> int:
  """A test fixture which enumerates graph label dimensionalities."""
  return request.param


@test.Fixture(
  scope="session",
  params=(False, True),
  namer=lambda x: f"log1p_graph_x:{str(x).lower()}",
)
def log1p_graph_x(request) -> bool:
  """Enumerate --log1p_graph_x values."""
  return request.param


@test.Fixture(
  scope="session",
  params=("zero", "constant", "constant_random", "random", "finetune", "none"),
  namer=lambda x: f"inst2vec_embeddings:{str(x).lower()}",
)
def node_text_embedding_type(request):
  return request.param


@test.Fixture(
  scope="session",
  params="none constant edge_count data_flow_max_steps label_convergence".split(),
  namer=lambda x: f"unroll_strategy:{str(x).lower()}",
)
def unroll_strategy(request) -> str:
  return request.param


@test.Fixture(
  scope="session",
  params=(["2", "2", "2", "2"], ["10"]),
  namer=lambda x: f"layer_timesteps:{','.join(str(y) for y in x)}",
)
def layer_timesteps(request) -> List[str]:
  return request.param


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("node_y_db"),
)
def node_y_graph_db(
  request, node_y_dimensionality: int,
) -> graph_tuple_database.Database:
  """A test fixture which yields a graph database with 256 OpenCL IR entries."""
  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, request.param
  ) as db:
    random_graph_tuple_database_generator.PopulateDatabaseWithRandomGraphTuples(
      db,
      graph_count=50,
      node_y_dimensionality=node_y_dimensionality,
      node_x_dimensionality=2,
      graph_y_dimensionality=0,
      with_data_flow=True,
      split_count=3,
    )
    yield db


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_y_db"),
)
def graph_y_graph_db(
  request, graph_y_dimensionality: int,
) -> graph_tuple_database.Database:
  """A test fixture which yields a graph database with 256 OpenCL IR entries."""
  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, request.param
  ) as db:
    random_graph_tuple_database_generator.PopulateDatabaseWithRandomGraphTuples(
      db,
      graph_count=50,
      node_x_dimensionality=2,
      node_y_dimensionality=0,
      graph_x_dimensionality=2,
      graph_y_dimensionality=graph_y_dimensionality,
      with_data_flow=True,
      split_count=3,
    )
    yield db
