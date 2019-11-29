"""Unit tests for //deeplearning/ml4pl/graphs/unlabelled:unlabelled_graph_database."""
import pathlib

from deeplearning.ml4pl.graphs import programl_pb2
from deeplearning.ml4pl.graphs.migrate import networkx_to_protos
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.graphs.unlabelled.cdfg import random_cdfg_generator
from labm8.py import app
from labm8.py import decorators
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="function")
def db(tempdir: pathlib.Path) -> unlabelled_graph_database.Database:
  """Fixture that returns an empty sqlite database."""
  yield unlabelled_graph_database.Database(f"sqlite:///{tempdir}/empty_db.db")


def graph_proto() -> programl_pb2.ProgramGraph:
  """Return a basically-empty graph proto."""
  return programl_pb2.ProgramGraph(
    node=[
      programl_pb2.Node(
        type=programl_pb2.Node.STATEMENT, text="root", encoded=0,
      ),
      programl_pb2.Node(
        type=programl_pb2.Node.IDENTIFIER, text="%1", encoded=1,
      ),
    ],
    edge=[
      programl_pb2.Edge(
        type=programl_pb2.Edge.DATA, source_node=0, destination_node=1,
      )
    ],
  )


@decorators.loop_for(seconds=10)
def test_fuzz_ProgramGraph_Create(db: unlabelled_graph_database.Database):
  """Fuzz the networkx -> proto conversion using randomly generated graphs."""
  g = random_cdfg_generator.FastCreateRandom()
  proto = networkx_to_protos.NetworkXGraphToProgramGraphProto(g)

  with db.Session(commit=True) as session:
    session.add(
      unlabelled_graph_database.ProgramGraph.Create(proto, "split", 0)
    )


# def test_benchmark_Graph_CreateFromNetworkX(benchmark, graph: nx.MultiDiGraph):
#   """Benchmark CreateFromNetworkX()."""
#   benchmark(graph_database.GraphMeta.CreateFromNetworkX, graph)


if __name__ == "__main__":
  test.Main()
