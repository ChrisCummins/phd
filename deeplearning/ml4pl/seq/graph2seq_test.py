"""Unit tests for //deeplearning/ml4pl/seq:lexer."""
import random
import string
from typing import Set

import sqlalchemy as sql

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.labelled.devmap import make_devmap_dataset
from deeplearning.ml4pl.ir import ir_database
from deeplearning.ml4pl.seq import graph2seq
from deeplearning.ml4pl.seq import ir2seq
from deeplearning.ml4pl.testing import random_graph_tuple_database_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


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
def graph_db(request) -> graph_tuple_database.Database:
  yield from testing_databases.YieldDatabase(
    graph_tuple_database.Database, request.param
  )


def CreateRandomString(min_length: int = 1, max_length: int = 1024) -> str:
  """Generate a random string."""
  return "".join(
    random.choice(string.ascii_lowercase)
    for _ in range(random.randint(min_length, max_length))
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
) -> graph_tuple_database.Database:
  """A test fixture which yields a graph database with 256 OpenCL IR entries."""
  with testing_databases.DatabaseContext(
    graph_tuple_database.Database, request.param
  ) as db:
    rows = []
    # Create random rows using OpenCL relpaths.
    for i, relpath in enumerate(opencl_relpaths):
      graph_tuple = (
        random_graph_tuple_database_generator.CreateRandomGraphTuple()
      )
      graph_tuple.ir_id = i + 1
      graph_tuple.id = len(opencl_relpaths) - i
      rows.append(graph_tuple)

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


@test.Fixture(scope="session", params=(None, 256))
def cache_size(request) -> int:
  """A test fixture to enumerate cache sizes."""
  return request.param


@test.Fixture(
  scope="function",
  params=(
    graph2seq.GraphEncoder,
    graph2seq.StatementEncoder,
    graph2seq.IdentifierEncoder,
  ),
)
def graph2seq_encoder(
  ir2seq_encoder: ir2seq.EncoderBase,
  populated_graph_db: graph_tuple_database.Database,
  cache_size: int,
) -> graph2seq.EncoderBase:
  """Test fixture that returns a graph encoder."""
  return graph2seq.GraphEncoder(
    populated_graph_db, ir2seq_encoder, cache_size=cache_size
  )


@decorators.loop_for(seconds=10)
def test_fuzz_Encode(graph2seq_encoder: ir2seq.EncoderBase):
  """Fuzz the encoder."""
  with graph2seq_encoder.graph_db.Session() as session:
    # Load a random collection of graphs. Note the joined load that is required
    # by StatementEncoder and IdentifierEncoder.
    graphs = (
      session.query(graph_tuple_database.GraphTuple)
      .order_by(graph2seq_encoder.graph_db.Random())
      .options(sql.orm.joinedload(graph_tuple_database.GraphTuple.data))
      .limit(random.randint(1, 200))
      .all()
    )
    # Sanity check that graphs are returned.
    assert graphs

  encodeds = graph2seq_encoder.Encode(graphs)

  assert len(encodeds) == len(graphs)


if __name__ == "__main__":
  test.Main()
