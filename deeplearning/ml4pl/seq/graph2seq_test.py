"""Unit tests for //deeplearning/ml4pl/seq:lexer."""
import random
import string
from typing import Set

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.labelled.devmap import make_devmap_dataset
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.ir import ir_database
from deeplearning.ml4pl.seq import graph2seq
from deeplearning.ml4pl.seq import ir2seq
from deeplearning.ml4pl.testing import random_graph_tuple_database_generator
from deeplearning.ml4pl.testing import (
  random_unlabelled_graph_database_generator,
)
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


###############################################################################
# Utility code.
###############################################################################


def CreateRandomString(min_length: int = 1, max_length: int = 1024) -> str:
  """Generate a random string."""
  return "".join(
    random.choice(string.ascii_lowercase)
    for _ in range(random.randint(min_length, max_length))
  )


###############################################################################
# Fixtures.
###############################################################################


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("ir_db"),
)
def ir_db(request) -> ir_database.Database:
  """A test fixture which yields an empty graph proto database."""
  yield from testing_databases.YieldDatabase(
    ir_database.Database, request.param
  )


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_db"),
)
def proto_db(request) -> unlabelled_graph_database.Database:
  yield from testing_databases.YieldDatabase(
    unlabelled_graph_database.Database, request.param
  )


@test.Fixture(scope="session")
def opencl_relpaths() -> Set[str]:
  opencl_df = make_devmap_dataset.MakeGpuDataFrame(
    opencl_device_mapping_dataset.OpenClDeviceMappingsDataset().df,
    "amd_tahiti_7970",
  )
  return set(opencl_df.relpath.values)


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("graph_db"),
)
def populated_graph_db(
  request, opencl_relpaths: Set[str]
) -> unlabelled_graph_database.Database:
  """A test fixture which yields a graph database with 256 OpenCL IR entries."""
  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, request.param
  ) as db:
    rows = []
    # Create random rows using OpenCL relpaths.
    for i, relpath in enumerate(opencl_relpaths):
      row = random_graph_tuple_database_generator.CreateRandomGraphTuple()
      row.ir_id = i + 1
      row.id = len(opencl_relpaths) - i
      rows.append(row)

    with db.Session(commit=True) as session:
      session.add_all(rows)

    yield db


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("proto_db"),
)
def populated_proto_db(
  request, opencl_relpaths: Set[str]
) -> unlabelled_graph_database.Database:
  """A test fixture which yields a graph database with 256 OpenCL IR entries."""
  with testing_databases.DatabaseContext(
    unlabelled_graph_database.Database, request.param
  ) as db:
    rows = []
    # Create random rows using OpenCL relpaths.
    for i, relpath in enumerate(opencl_relpaths):
      row = (
        random_unlabelled_graph_database_generator.CreateRandomProgramGraph()
      )
      row.ir_id = i + 1
      row.id = len(opencl_relpaths) - i
      rows.append(row)

    with db.Session(commit=True) as session:
      session.add_all(rows)

    yield db


@test.Fixture(
  scope="session",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("ir_db"),
)
def populated_ir_db(request, opencl_relpaths: Set[str]) -> ir_database.Database:
  """A test fixture which yields an IR database with 256 OpenCL entries."""
  with testing_databases.DatabaseContext(
    ir_database.Database, request.param
  ) as db:
    rows = []
    # Create IRs using OpenCL relpaths.
    for i, relpath in enumerate(opencl_relpaths):
      ir = ir_database.IntermediateRepresentation.CreateFromText(
        source="pact17_opencl_devmap",
        relpath=relpath,
        source_language=ir_database.SourceLanguage.OPENCL,
        type=ir_database.IrType.LLVM_6_0,
        cflags="",
        text=CreateRandomString(),
      )
      ir.id = i + 1
      rows.append(ir)

    with db.Session(commit=True) as session:
      session.add_all(rows)

    yield db


@test.Fixture(
  scope="function",
  params=(ir2seq.LlvmEncoder, ir2seq.OpenClEncoder, ir2seq.Inst2VecEncoder),
)
def ir2seq_encoder(
  request, populated_ir_db: ir_database.Database
) -> ir2seq.EncoderBase:
  """Test fixture an encoder with IR IDs in range [1,256]."""
  return request.param(populated_ir_db)


@test.Fixture(
  scope="session",
  params=(None, 256),
  names=("cache_size:default", "cache_size:256"),
)
def cache_size(request) -> int:
  """A test fixture to enumerate cache sizes."""
  return request.param


###############################################################################
# Tests.
###############################################################################


@decorators.loop_for(seconds=3)
def test_fuzz_GraphEncoder(
  ir2seq_encoder: ir2seq.EncoderBase,
  populated_graph_db: graph_tuple_database.Database,
  cache_size: int,
):
  """Fuzz the encoder."""
  graph2seq_encoder = graph2seq.GraphEncoder(
    populated_graph_db, ir2seq_encoder, cache_size
  )

  with graph2seq_encoder.graph_db.Session() as session:
    # Load a random collection of graphs.
    graphs = (
      session.query(graph_tuple_database.GraphTuple)
      .order_by(graph2seq_encoder.graph_db.Random())
      .limit(random.randint(1, 200))
      .all()
    )
    # Sanity check that graphs are returned.
    assert graphs

  encoded = graph2seq_encoder.Encode(graphs)

  assert len(encoded) == len(graphs)


@decorators.loop_for(seconds=3)
def test_fuzz_StatementEncoder(
  ir2seq_encoder: ir2seq.EncoderBase,
  populated_proto_db: unlabelled_graph_database.Database,
  populated_graph_db: graph_tuple_database.Database,
  cache_size: int,
):
  """Fuzz the encoder."""
  graph2seq_encoder = graph2seq.StatementEncoder(
    populated_graph_db, populated_proto_db, ir2seq_encoder, cache_size
  )

  with graph2seq_encoder.graph_db.Session() as session:
    # Load a random collection of graphs.
    graphs = (
      session.query(graph_tuple_database.GraphTuple)
      .order_by(graph2seq_encoder.graph_db.Random())
      .limit(random.randint(1, 200))
      .all()
    )
    # Sanity check that graphs are returned.
    assert graphs

  encoded = graph2seq_encoder.Encode(graphs)

  assert len(encoded) == len(graphs)


if __name__ == "__main__":
  test.Main()
