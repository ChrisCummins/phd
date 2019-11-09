"""Unit tests for //deeplearning/ml4pl/graphs/labelled/graph_batcher."""
import pathlib
import pickle
import typing

import numpy as np
import pytest
from labm8 import app
from labm8 import test

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs.labelled import make_data_flow_analysis_dataset

FLAGS = app.FLAGS


@pytest.fixture(scope='function')
def db(tempdir: pathlib.Path):
  yield graph_database.Database(f'sqlite:///{tempdir}/db1')


@pytest.fixture(scope='function')
def db2(tempdir: pathlib.Path):
  yield graph_database.Database(f'sqlite:///{tempdir}/db2')


@pytest.fixture(scope='function')
def db3(tempdir: pathlib.Path):
  yield graph_database.Database(f'sqlite:///{tempdir}/db3')


@pytest.fixture(scope='function')
def db4(tempdir: pathlib.Path):
  yield graph_database.Database(f'sqlite:///{tempdir}/db4')


def test_GetAnnotatedGraphGenerators_unknown_analysis():
  with pytest.raises(app.UsageError):
    make_data_flow_analysis_dataset.GetAnnotatedGraphGenerators('foo')


def test_GetAnnotatedGraphGenerators_with_requested_analyses():
  """Test requesting analyses by name."""
  annotators = make_data_flow_analysis_dataset.GetAnnotatedGraphGenerators(
      'reachability', 'domtree')
  annotator_names = {a.name for a in annotators}

  assert annotator_names == {'reachability', 'domtree'}


def MakeGraphMeta(bytecode_id: int):
  """Construct a graph meta, where only bytecode_id is important."""
  return graph_database.GraphMeta(
      group='foo',
      bytecode_id=bytecode_id,
      source_name='foo',
      relpath='foo',
      language='c',
      node_count=10,
      edge_count=100,
      node_type_count=1,
      edge_type_count=6,
      edge_position_max=5,
      loop_connectedness=0,
      undirected_diameter=0,
      graph=graph_database.Graph(pickled_data=pickle.dumps('123')))


def AddGraphMetas(db_: graph_database.Database, bytecode_ids: typing.List[int]):
  with db_.Session(commit=True) as session:
    session.add_all([MakeGraphMeta(id) for id in bytecode_ids])
  return db_


# GetBytecodesToProcessForOutput() tests.


def test_GetBytecodesToProcessForOutput_empty_db(db: graph_database.Database):
  output_db = AddGraphMetas(db, [])
  assert make_data_flow_analysis_dataset.GetBytecodesToProcessForOutput(
      {1, 2, 3, 5, 10}, output_db) == {1, 2, 3, 5, 10}


def test_GetBytecodesToProcessForOutput_empty_db(db: graph_database.Database):
  output_db = AddGraphMetas(db, [1, 2, 3, 4, 6])
  assert make_data_flow_analysis_dataset.GetBytecodesToProcessForOutput(
      {1, 2, 3, 5, 10}, output_db) == {5, 10}


# GetBytecodeIdsToProcess() tests.


def test_GetBytecodeIdsToProcess_empty_output_databases(db, db2, db3, db4):
  input_db = AddGraphMetas(db, [1, 2, 3, 5, 10])
  output_dbs = [db2, db3, db4]

  all_ids, ids_by_output = make_data_flow_analysis_dataset.GetBytecodeIdsToProcess(
      input_db, output_dbs)

  assert np.array_equal(all_ids, [1, 2, 3, 5, 10])
  assert ids_by_output.shape == (3, 5)
  for row in ids_by_output:
    assert np.array_equal(row, [1, 2, 3, 5, 10])


def test_GetBytecodeIdsToProcess_with_some_outputs(db, db2, db3, db4):
  input_db = AddGraphMetas(db, [1, 2, 3, 5, 10])
  output_dbs = [
      AddGraphMetas(db2, [
          1,
      ]),
      AddGraphMetas(db3, [
          1,
          2,
          3,
          5,
      ]),
      AddGraphMetas(db4, [
          1,
          5,
      ]),
  ]

  all_ids, ids_by_output = make_data_flow_analysis_dataset.GetBytecodeIdsToProcess(
      input_db, output_dbs)

  assert np.array_equal(sorted(all_ids), [2, 3, 5, 10])
  assert ids_by_output.shape == (3, 4)
  # Note that results are ordered by frequency, from least to most.
  assert np.array_equal(ids_by_output[0], [5, 2, 3, 10])
  assert np.array_equal(ids_by_output[1], [0, 0, 0, 10])
  assert np.array_equal(ids_by_output[2], [0, 2, 3, 10])


def test_DataFlowAnalysisGraphExporter(db, db2, db3, db4):
  input_db = AddGraphMetas(db, [1, 2, 3, 5, 10])
  output_dbs = [AddGraphMetas(db2, [1, 2]), db3, db4]

  outputs = [
      make_data_flow_analysis_dataset.Output(
          annotator=make_data_flow_analysis_dataset.GetAnnotatedGraphGenerators(
              'reachability')[0],
          db=output_dbs[0]),
      make_data_flow_analysis_dataset.Output(
          annotator=make_data_flow_analysis_dataset.GetAnnotatedGraphGenerators(
              'liveness')[0],
          db=output_dbs[1]),
      make_data_flow_analysis_dataset.Output(
          annotator=make_data_flow_analysis_dataset.GetAnnotatedGraphGenerators(
              'domtree')[0],
          db=output_dbs[2]),
  ]

  make_data_flow_analysis_dataset.DataFlowAnalysisGraphExporter(outputs)(
      input_db, output_dbs)

  for output_db in output_dbs:
    with output_db.Session() as session:
      bytecode_ids = [
          row.bytecode_id
          for row in session.query(graph_database.GraphMeta.bytecode_id)
      ]
      assert sorted(bytecode_ids) == [1, 2, 3, 5, 10]

  # Running it a second time should make no changes.
  make_data_flow_analysis_dataset.DataFlowAnalysisGraphExporter(outputs)(
      input_db, output_dbs)

  for output_db in output_dbs:
    with output_db.Session() as session:
      bytecode_ids = [
          row.bytecode_id
          for row in session.query(graph_database.GraphMeta.bytecode_id)
      ]
      assert sorted(bytecode_ids) == [1, 2, 3, 5, 10]


if __name__ == '__main__':
  test.Main()
