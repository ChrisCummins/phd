"""Test fixtures for graph databases."""
import pathlib
import pickle

import numpy as np
import pytest

from deeplearning.ml4pl.graphs import graph_database
from labm8.py import test

FLAGS = test.FLAGS


@pytest.fixture(scope="function")
def empty_graph_db(tempdir: pathlib.Path) -> graph_database.Database:
  """Fixture that returns an sqlite database."""
  yield graph_database.Database(f"sqlite:///{tempdir}/graphs.db")


@pytest.fixture(scope="function")
def graph_db_512(
  empty_graph_db: graph_database.Database,
) -> graph_database.Database:
  """Fixture which returns a database with 512 graphs, indexed by node_count."""

  def _MakeGraphMeta(i):
    return graph_database.GraphMeta(
      group="train",
      bytecode_id=1,
      source_name="foo",
      relpath="bar",
      language="c",
      node_count=i,
      edge_count=2,
      edge_position_max=0,
      loop_connectedness=0,
      undirected_diameter=0,
      data_flow_max_steps_required=i,
      graph=graph_database.Graph(
        pickled_data=pickle.dumps(np.ones(200000 // 4) * i)  # ~200KB of data
      ),
    )

  with empty_graph_db.Session(commit=True) as s:
    s.add_all([_MakeGraphMeta(i) for i in range(512)])

  return empty_graph_db
