"""Unit tests for //deeplearning/ml4pl/models/lstm:bytecode2seq."""
import pathlib

import numpy as np
import pandas as pd

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.ml4pl.bytecode import bytecode_database
from deeplearning.ml4pl.graphs.labelled.devmap import make_devmap_dataset
from deeplearning.ml4pl.seq import ir2seq
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


@test.Fixture(scope="module")
def opencl_dataset_df() -> pd.DataFrame:
  """Test fixture which yields a dataframe of the OpenCL dataset, complete with
  relpath column.
  """
  dataset = opencl_device_mapping_dataset.OpenClDeviceMappingsDataset()
  return make_devmap_dataset.MakeGpuDataFrame(dataset.df, "amd_tahiti_7970")


@test.Fixture(scope="function")
def bytecode_db(
  tempdir: pathlib.Path,
  opencl_dataset_df: opencl_device_mapping_dataset.OpenClDeviceMappingsDataset,
) -> bytecode_database.Database:
  """Test fixture which returns a bytecode datbase.

  This uses the relpaths from the OpenCL devmap dataset so that OpenClEncoder
  can lookup the relpath -> bytecode IDs.
  """
  # There are duplicate relpaths in the dataset.
  relpaths = set([row["relpath"] for _, row in opencl_dataset_df.iterrows()])
  db = bytecode_database.Database(f"sqlite:///{tempdir}/bytecodes.db")
  with db.Session(commit=True) as session:
    session.add_all(
      [
        bytecode_database.LlvmBytecode(
          source_name="pact17_opencl_devmap",
          relpath=relpath,
          language="c",
          cflags="",
          charcount=100,
          linecount=10,
          bytecode="1234",
          clang_returncode=0,
          error_message="",
          bytecode_sha1="",
        )
        for relpath in relpaths
      ]
    )
  return db


@test.Parametrize(
  "encoder_class",
  [ir2seq.BytecodeEncoder, ir2seq.OpenClEncoder, ir2seq.Inst2VecEncoder,],
)
def test_Encode(bytecode_db: bytecode_database.Database, encoder_class):
  FLAGS.bytecode_db = lambda: bytecode_db

  encoder = encoder_class()
  encoded = encoder.Encode([1, 1, 2, 3, 4])
  assert len(encoded) == 5
  # The first two requested bytecodes are the same.
  assert np.array_equal(encoded[0], encoded[1])


if __name__ == "__main__":
  test.Main()
