"""Unit tests for //deeplearning/ml4pl/seq:lexer."""
import random
import string
from typing import Set

import numpy as np

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.ml4pl.graphs.labelled.devmap import make_devmap_dataset
from deeplearning.ml4pl.ir import ir_database
from deeplearning.ml4pl.seq import ir2seq
from deeplearning.ml4pl.testing import testing_databases
from labm8.py import decorators
from labm8.py import test

FLAGS = test.FLAGS


@test.Fixture(
  scope="function",
  params=testing_databases.GetDatabaseUrls(),
  namer=testing_databases.DatabaseUrlNamer("ir_db"),
)
def ir_db(request) -> ir_database.Database:
  """A test fixture which yields an empty graph proto database."""
  yield from testing_databases.YieldDatabase(
    ir_database.Database, request.param
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


@test.Fixture(
  scope="function",
  params=(ir2seq.LlvmEncoder, ir2seq.OpenClEncoder, ir2seq.Inst2VecEncoder),
)
def encoder(
  request, populated_ir_db: ir_database.Database
) -> ir2seq.EncoderBase:
  """Test fixture an encoder with IR IDs in range [1,100]."""
  return request.param(populated_ir_db)


@decorators.loop_for(seconds=30)
def test_fuzz_Encode(encoder: ir2seq.EncoderBase):
  """Fuzz the encoder."""
  ids = [random.randint(1, 256) for _ in range(random.randint(1, 200))]

  encodeds = encoder.Encode(ids)

  assert len(encodeds) == len(ids)
  for encoded in encodeds:
    assert not np.where(encoded > encoder.vocabulary_size + 1)[0].size


if __name__ == "__main__":
  test.Main()
