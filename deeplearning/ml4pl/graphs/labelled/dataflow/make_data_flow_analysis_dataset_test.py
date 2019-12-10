"""Unit tests for //deeplearning/ml4pl/graphs/labelled/dataflow:make_data_flow_analysis_dataset."""
import pathlib
import pickle
import time
import typing

import networkx as nx
import numpy as np

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.labelled.dataflow import data_flow_graphs
from deeplearning.ml4pl.graphs.labelled.dataflow import (
  make_data_flow_analysis_dataset,
)
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS

# TODO(github.com/ChrisCummins/ProGraML/issues/2): Overhaul.
MODULE_UNDER_TEST = None


def test_TODO():
  pass


#
# class EternalAnnotator(data_flow_graphs.DataFlowGraphAnnotator):
#   """An annotator that takes ages."""
#
#   def RootNodeType(self) -> programl_pb2.Node.Type:
#     """Return the Node.Type enum for root nodes."""
#     return programl_pb2.Node.STATEMENT
#
#   def Annotate(
#     self, g: nx.MultiDiGraph, root_node: int
#   ) -> data_flow_graphs.DataFlowAnnotatedGraph:
#     """Annotate a networkx graph in-place."""
#     time.sleep(1000)
#
#
# class FailingAnnotator(data_flow_graphs.DataFlowGraphAnnotator):
#   """An annotator that fails."""
#
#   def RootNodeType(self) -> programl_pb2.Node.Type:
#     """Return the Node.Type enum for root nodes."""
#     return programl_pb2.Node.STATEMENT
#
#   def Annotate(
#     self, g: nx.MultiDiGraph, root_node: int
#   ) -> data_flow_graphs.DataFlowAnnotatedGraph:
#     """Annotate a networkx graph in-place."""
#     raise ValueError("oh noes!")
#
# @test.Fixture(scope="function")
# def db(tempdir: pathlib.Path):
#   yield graph_database.Database(f"sqlite:///{tempdir}/db1")
#
#
# @test.Fixture(scope="function")
# def db2(tempdir: pathlib.Path):
#   yield graph_database.Database(f"sqlite:///{tempdir}/db2")
#
#
# @test.Fixture(scope="function")
# def db3(tempdir: pathlib.Path):
#   yield graph_database.Database(f"sqlite:///{tempdir}/db3")
#
#
# @test.Fixture(scope="function")
# def db4(tempdir: pathlib.Path):
#   yield graph_database.Database(f"sqlite:///{tempdir}/db4")
#
#
# def test_GetAnnotatedGraphGenerators_unknown_analysis():
#   """Test that unknown analysis raises error."""
#   with test.Raises(app.UsageError):
#     make_data_flow_analysis_dataset.GetDataFlowGraphAnnotator("foo")
#
#
# def MakeGraphMeta(bytecode_id: int):
#   """Construct a graph meta, where only bytecode_id is important."""
#   return graph_database.GraphMeta(
#     group="foo",
#     bytecode_id=bytecode_id,
#     source_name="foo",
#     relpath="foo",
#     language="c",
#     node_count=10,
#     edge_count=100,
#     node_type_count=1,
#     edge_type_count=6,
#     edge_position_max=5,
#     loop_connectedness=0,
#     undirected_diameter=0,
#     graph=graph_database.Graph(pickled_data=pickle.dumps("123")),
#   )
#
#
# def AddGraphMetas(db_: graph_database.Database, bytecode_ids: typing.List[int]):
#   with db_.Session(commit=True) as session:
#     session.add_all([MakeGraphMeta(id) for id in bytecode_ids])
#   return db_
#
#
# # GetBytecodeIdsToProcess() tests.
#
#
# def test_GetBytecodeIdsToProcess_empty_output_databases(db, db2, db3, db4):
#   input_db = AddGraphMetas(db, [1, 2, 3, 5, 10])
#   output_dbs = [db2, db3, db4]
#
#   (
#     all_ids,
#     ids_by_output,
#   ) = make_data_flow_analysis_dataset.GetBytecodeIdsToProcess(
#     {1, 2, 3, 5, 10}, output_dbs, 16
#   )
#
#   assert np.array_equal(all_ids, [1, 2, 3, 5, 10])
#   assert ids_by_output.shape == (3, 5)
#   for row in ids_by_output:
#     assert np.array_equal(row, [1, 2, 3, 5, 10])
#
#
# def test_GetBytecodeIdsToProcess_with_some_outputs(db, db2, db3, db4):
#   input_db = AddGraphMetas(db, [1, 2, 3, 5, 10])
#   output_dbs = [
#     AddGraphMetas(db2, [1,]),
#     AddGraphMetas(db3, [1, 2, 3, 5,]),
#     AddGraphMetas(db4, [1, 5,]),
#   ]
#
#   (
#     all_ids,
#     ids_by_output,
#   ) = make_data_flow_analysis_dataset.GetBytecodeIdsToProcess(
#     {1, 2, 3, 5, 10}, output_dbs, 16
#   )
#
#   assert np.array_equal(sorted(all_ids), [2, 3, 5, 10])
#   assert ids_by_output.shape == (3, 4)
#   # Note that results are ordered by frequency, from least to most.
#   assert np.array_equal(ids_by_output[0], [5, 2, 3, 10])
#   assert np.array_equal(ids_by_output[1], [0, 0, 0, 10])
#   assert np.array_equal(ids_by_output[2], [0, 2, 3, 10])
#
#
# # ResilientAddUnique() tests.
#
#
# def test_ResilientAddUnique_empty_db(db: graph_database.Database):
#   just_done = [1, 1, 1, 2]
#   make_data_flow_analysis_dataset.ResilientAddUnique(
#     db, [MakeGraphMeta(i) for i in just_done], "foo"
#   )
#
#   with db.Session() as session:
#     assert session.query(graph_database.GraphMeta).count() == 4
#     assert (
#       session.query(graph_database.GraphMeta)
#       .filter(graph_database.GraphMeta.bytecode_id == 1)
#       .count()
#       == 3
#     )
#     assert (
#       session.query(graph_database.GraphMeta)
#       .filter(graph_database.GraphMeta.bytecode_id == 2)
#       .count()
#       == 1
#     )
#
#
# def test_ResilientAddUnique_with_dupes(db: graph_database.Database):
#   """Test that ResilientAddUnique() ignores bytecodes that already exist."""
#   already_done = [1, 1, 3, 3]
#   just_done = [1, 1, 1, 2, 2, 2]
#
#   db = AddGraphMetas(db, already_done)
#   make_data_flow_analysis_dataset.ResilientAddUnique(
#     db, [MakeGraphMeta(i) for i in just_done], "foo"
#   )
#
#   with db.Session() as session:
#     assert session.query(graph_database.GraphMeta).count() == 7
#     assert (
#       session.query(graph_database.GraphMeta)
#       .filter(graph_database.GraphMeta.bytecode_id == 1)
#       .count()
#       == 2
#     )
#     assert (
#       session.query(graph_database.GraphMeta)
#       .filter(graph_database.GraphMeta.bytecode_id == 2)
#       .count()
#       == 3
#     )
#     assert (
#       session.query(graph_database.GraphMeta)
#       .filter(graph_database.GraphMeta.bytecode_id == 3)
#       .count()
#       == 2
#     )
#
#
# # DataFlowAnalysisGraphExporter() tests.
#
#
# @test.XFail(reason="TODO(github.com/ChrisCummins/ProGraML/issues/2): Fix me.")
# def test_DataFlowAnalysisGraphExporter_integration_test(db, db2, db3, db4):
#   """Test end-to-end dataset export with three annotators."""
#   all_bytecode_ids = [1, 2, 3, 5, 10]
#   already_done = [1, 2]
#
#   input_db = AddGraphMetas(db, all_bytecode_ids)
#   output_dbs = [AddGraphMetas(db2, already_done), db3, db4]
#
#   def MakeOutput(annotator_name, db_):
#     """Generate an output from the given name and database."""
#     return make_data_flow_analysis_dataset.Output(
#       annotator=make_data_flow_analysis_dataset.GetAnnotatedGraphGenerators(
#         annotator_name
#       )[0],
#       db=db_,
#     )
#
#   outputs = [
#     MakeOutput("reachability", output_dbs[0]),
#     MakeOutput("liveness", output_dbs[1]),
#     MakeOutput("domtree", output_dbs[2]),
#   ]
#
#   exporter = make_data_flow_analysis_dataset.DataFlowAnalysisGraphExporter(
#     outputs
#   )
#   exporter(input_db, output_dbs)
#
#   # Check that all databases have an entry.
#   for output_db in output_dbs:
#     with output_db.Session() as session:
#       bytecode_ids = [
#         row.bytecode_id
#         for row in session.query(graph_database.GraphMeta.bytecode_id)
#       ]
#       assert sorted(bytecode_ids) == [1, 2, 3, 5, 10]
#
#   # Running the exporter multiple times should yield no changes.
#   exporter(input_db, output_dbs)
#   exporter(input_db, output_dbs)
#
#   # Check that all databases have an entry.
#   for output_db in output_dbs:
#     with output_db.Session() as session:
#       bytecode_ids = [
#         row.bytecode_id
#         for row in session.query(graph_database.GraphMeta.bytecode_id)
#       ]
#       assert sorted(bytecode_ids) == [1, 2, 3, 5, 10]


if __name__ == "__main__":
  test.Main()
