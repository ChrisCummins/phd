"""Unit tests for //deeplearning/ml4pl/graphs/migrate:networkx_to_protos."""
import pathlib
import pickle
import typing

import networkx as nx

from deeplearning.ml4pl.graphs import graph_database
from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.migrate import networkx_to_protos
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.graphs.unlabelled.cdfg import random_cdfg_generator
from labm8.py import bazelutil
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS

GRAPH_PROTOS_ARCHIVE = bazelutil.DataArchive(
  "phd/deeplearning/ml4pl/testing/data/100_unlabelled_networkx_graphs.tar.bz2"
)

GRAPH_DB = bazelutil.DataArchive(
  "phd/deeplearning/ml4pl/testing/data/100_unlabelled_networkx_graphs.db.tar.bz2"
)


@test.Fixture(scope="function")
def valid_simple_graph() -> nx.MultiDiGraph:
  """A test fixture which returns a CDFG for the following pair of statements:

      a = b * c + g;
      d = b * c * e;
  """
  g = nx.MultiDiGraph()
  g.add_node("a1", type="identifier", name="a1", x=-1, function="a")
  g.add_node("a2", type="identifier", name="a2", x=-1, function="a")
  g.add_node("b", type="identifier", name="b", x=-1, function="a")
  g.add_node("c", type="identifier", name="c", x=-1, function="a")
  g.add_node("d1", type="identifier", name="d1", x=-1, function="a")
  g.add_node("d2", type="identifier", name="d2", x=-1, function="a")
  g.add_node("e", type="identifier", name="e", x=-1, function="a")
  g.add_node("g", type="identifier", name="g", x=-1, function="a")

  g.add_node(
    "s0",
    type="statement",
    text="<ID> = mul <ID> <ID>",
    original_text="%a1 = div %b %c",
    x=-1,
  )
  g.add_node(
    "s1",
    type="statement",
    text="<ID> = mul <ID> <ID>",
    original_text="%d1 = div %b %c",
    x=-1,
  )
  g.add_node(
    "s2",
    type="statement",
    text="<ID> = add <ID> <ID>",
    original_text="%a2 = add %a1 %g",
    x=-1,
  )
  g.add_node(
    "s3",
    type="statement",
    text="<ID> = mul <ID> <ID>",
    original_text="%d2 = mul %d1 %e",
    x=-1,
  )

  g.add_node("root", name="root", type="magic")
  g.add_edge("root", "a1", flow="call")
  g.add_edge("root", "d1", flow="call")

  g.add_edge("s0", "a1", flow="data", position=0)
  g.add_edge("b", "s0", flow="data", position=0)
  g.add_edge("c", "s0", flow="data", position=1)
  g.add_edge("s1", "d1", flow="data", position=0)
  g.add_edge("b", "s1", flow="data", position=0)
  g.add_edge("c", "s1", flow="data", position=1)

  g.add_edge("a1", "s2", flow="data", position=0)
  g.add_edge("g", "s2", flow="data", position=1)
  g.add_edge("s2", "a2", flow="data", position=0)

  g.add_edge("d1", "s3", flow="data", position=0)
  g.add_edge("e", "s3", flow="data", position=1)
  g.add_edge("s3", "d2", flow="data", position=0)
  return g


def test_NetworkXGraphToProgramGraphProto_returns_proto(
  valid_simple_graph: nx.MultiDiGraph,
):
  """Test that a proto is returned."""
  proto = networkx_to_protos.NetworkXGraphToProgramGraphProto(
    valid_simple_graph
  )
  assert isinstance(proto, programl_pb2.ProgramGraph)


def ReadPickledNetworkxGraphs() -> typing.Iterable[
  typing.Tuple[str, nx.MultiDiGraph]
]:
  """Read the pickled networkx graphs."""
  with GRAPH_PROTOS_ARCHIVE as pickled_dir:
    for path in pickled_dir.iterdir():
      with open(path, "rb") as f:
        yield path.name, pickle.load(f)


@test.Fixture(scope="function", params=list(ReadPickledNetworkxGraphs()))
def pickled_networkx_graph(request) -> int:
  """A parametrized test fixture."""
  return request.param


def test_NetworkXGraphToProgramGraphProto_random_100(
  pickled_networkx_graph: typing.Tuple[str, nx.MultiDiGraph]
):
  """Test networkx -> proto conversion over the 100 test graphs."""
  name, g = pickled_networkx_graph
  proto = networkx_to_protos.NetworkXGraphToProgramGraphProto(g)

  assert isinstance(proto, programl_pb2.ProgramGraph)
  # The graph should have the same number of nodes and edges.
  assert len(proto.node) == g.number_of_nodes()
  assert len(proto.edge) == g.number_of_edges()
  # Check the root node.
  assert proto.node[0].text == "root"


@test.Fixture(scope="function")
def populated_graph_db() -> graph_database.Database:
  with GRAPH_DB as db_dir:
    assert (db_dir / "100_unlabelled_networkx_graphs.db").is_file()
    yield graph_database.Database(
      f"sqlite:///{db_dir}/100_unlabelled_networkx_graphs.db"
    )


@test.Fixture(scope="function")
def empty_output_db(
  tempdir: pathlib.Path,
) -> unlabelled_graph_database.Database:
  yield unlabelled_graph_database.Database(f"sqlite:///{tempdir}/output_db.db")


@test.Parametrize("nproc", (1, 2, 4))
def test_MigrateGraphDatabase_random_100(
  populated_graph_db: graph_database.Database,
  empty_output_db: unlabelled_graph_database.Database,
  nproc: int,
):
  """Run database migration on a test database of 100 "real" networkx graphs."""
  networkx_to_protos.MigrateGraphDatabase(
    populated_graph_db, empty_output_db, nproc
  )


@decorators.loop_for(seconds=10)
def test_fuzz_NetworkXGraphToProgramGraphProto():
  """Fuzz the networkx -> proto conversion using randomly generated graphs."""
  g = random_cdfg_generator.FastCreateRandom()
  proto = networkx_to_protos.NetworkXGraphToProgramGraphProto(g)

  assert isinstance(proto, programl_pb2.ProgramGraph)
  # The graph should have the same number of nodes and edges.
  assert len(proto.node) == g.number_of_nodes()
  assert len(proto.edge) == g.number_of_edges()
  # Check the root node.
  assert proto.node[0].text == "root"


if __name__ == "__main__":
  test.Main()
