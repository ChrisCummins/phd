"""Unit tests for //deeplearning/ml4pl/graphs/labelled/devmap:make_devmap_dataset."""
import random
import string
from typing import Set

import sqlalchemy as sql

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.ml4pl.graphs.labelled import graph_tuple_database
from deeplearning.ml4pl.graphs.labelled.devmap import make_devmap_dataset
from deeplearning.ml4pl.graphs.unlabelled import unlabelled_graph_database
from deeplearning.ml4pl.ir import ir_database
from deeplearning.ml4pl.testing import random_programl_generator
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import progress
from labm8.py import test

FLAGS = test.FLAGS


@test.Fixture(scope="function", params=testing_databases.GetDatabaseUrls())
def ir_db(request) -> ir_database.Database:
  """A test fixture which yields an empty graph proto database."""
  yield from testing_databases.YieldDatabase(
    ir_database.Database, request.param
  )


@test.Fixture(scope="function", params=testing_databases.GetDatabaseUrls())
def proto_db(request) -> unlabelled_graph_database.Database:
  yield from testing_databases.YieldDatabase(
    unlabelled_graph_database.Database, request.param
  )


@test.Fixture(scope="function", params=testing_databases.GetDatabaseUrls())
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


@test.Fixture(scope="function")
def populated_proto_db(
  proto_db: graph_tuple_database.Database, opencl_relpaths: Set[str]
) -> graph_tuple_database.Database:
  """A test fixture which yields a graph database with 256 OpenCL IR entries."""
  rows = []
  # Create random rows using OpenCL relpaths.
  for i, relpath in enumerate(opencl_relpaths):
    proto = unlabelled_graph_database.ProgramGraph.Create(
      proto=random_programl_generator.CreateRandomProto(), ir_id=i + 1
    )
    proto.id = len(opencl_relpaths) - i
    rows.append(proto)

  with proto_db.Session(commit=True) as session:
    session.add_all(rows)

  return proto_db


@test.Fixture(scope="function")
def populated_ir_db(
  ir_db: ir_database.Database, opencl_relpaths: Set[str]
) -> ir_database.Database:
  """A test fixture which yields an IR database with 256 OpenCL entries."""
  rows = []
  # Create random rows using OpenCL relpaths.
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

  with ir_db.Session(commit=True) as session:
    session.add_all(rows)

  return ir_db


@test.Parametrize("gpu", ("amd_tahiti_7970", "nvidia_gtx_960"))
def test_MakeOpenClDevmapDataset(
  populated_ir_db: ir_database.Database,
  populated_proto_db: unlabelled_graph_database.Database,
  graph_db: graph_tuple_database.Database,
  gpu: str,
):
  """Test that the expected number of graph tuples are generated."""
  job = make_devmap_dataset.MakeOpenClDevmapDataset(
    ir_db=populated_ir_db,
    proto_db=populated_proto_db,
    graph_db=graph_db,
    gpu=gpu,
  )
  progress.Run(job)
  with graph_db.Session() as session:
    assert (
      session.query(sql.func.count(graph_tuple_database.GraphTuple.id)).scalar()
      >= 256
    )
    # Check that there are 2-D node features.
    assert (
      session.query(graph_tuple_database.GraphTuple.node_x_dimensionality)
      .first()
      .node_x_dimensionality
      == 2
    )


if __name__ == "__main__":
  test.Main()
