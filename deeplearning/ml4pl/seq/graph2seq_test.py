"""Unit tests for //deeplearning/ml4pl/models/lstm:graph2seq."""
import pathlib
import pickle

import networkx as nx

from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.seq import graph2seq
from deeplearning.ml4pl.seq import ir2seq
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def bytecode_db(tempdir: pathlib.Path) -> bytecode_database.Database:
  db = bytecode_database.Database(f"sqlite:///{tempdir}/bytecodes.db")
  with db.Session(commit=True) as session:
    session.add(
      bytecode_database.LlvmBytecode(
        id=1,
        source_name="",
        relpath="",
        language="",
        cflags="",
        charcount=len("Hello, world!"),
        linecount=1,
        bytecode="Hello, world!",
        clang_returncode=0,
        error_message="",
        bytecode_sha1="",
      )
    )
  yield db


@test.Fixture(scope="function")
def graph_db(tempdir: pathlib.Path) -> graph_database.Database:
  db = graph_database.Database(f"sqlite:///{tempdir}/graphs.db")
  # Add a program graph, which is used by GraphToBytecodeGroupingsEncoder.
  g = nx.MultiDiGraph()
  g.add_node("root", type="magic")
  g.add_node("a", type="statement", function="a", original_text="a")
  g.add_node("%1", type="identifier", function="a", original_text="%1")
  g.add_edge("root", "a", flow="call")
  g.add_edge("%1", "a", flow="data")
  with db.Session(commit=True) as session:
    session.add(
      graph_database.GraphMeta(
        id=1,
        bytecode_id=1,
        group="",
        source_name="",
        relpath="",
        language="",
        node_count=0,
        edge_count=0,
        edge_position_max=0,
        loop_connectedness=0,
        undirected_diameter=0,
        graph=graph_database.Graph(pickled_data=pickle.dumps(g)),
      )
    )
  yield db


def test_GraphToBytecodeEncoder_Encode(
  graph_db: graph_database.Database, bytecode_db: bytecode_database.Database
):
  FLAGS.bytecode_db = lambda: bytecode_db
  encoder = graph2seq.GraphToBytecodeEncoder(graph_db, ir2seq.BytecodeEncoder())
  encoded = encoder.Encode([1])
  assert len(encoded) == 1
  assert len(encoded[0])


@test.Parametrize("group_by", ("statement", "identifier"))
def test_GraphToBytecodeGroupingsEncoder_Encode(
  graph_db: graph_database.Database,
  bytecode_db: bytecode_database.Database,
  group_by: str,
):
  FLAGS.bytecode_db = lambda: bytecode_db
  FLAGS.unlabelled_graph_db = lambda: graph_db

  encoder = graph2seq.GraphToBytecodeGroupingsEncoder(
    graph_db, ir2seq.BytecodeEncoder(), group_by
  )

  encoded_sequences, segment_ids, node_masks = encoder.Encode([1])

  assert 1 in encoded_sequences
  assert 1 in segment_ids
  assert 1 in node_masks

  assert len(encoded_sequences) == 1
  assert len(segment_ids) == 1
  assert len(node_masks) == 1

  assert len(encoded_sequences[1])
  assert len(segment_ids[1])
  assert len(node_masks[1])


if __name__ == "__main__":
  test.Main()
