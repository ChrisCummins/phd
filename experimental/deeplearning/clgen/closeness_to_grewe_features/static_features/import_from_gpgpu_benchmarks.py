"""Import kernels from GPGPU benchmark suites."""
import hashlib
import typing

import progressbar

from datasets.opencl.device_mapping import opencl_device_mapping_dataset
from deeplearning.deeptune.opencl.heterogeneous_mapping.models import ncc
from experimental.deeplearning.clgen.closeness_to_grewe_features import (
  grewe_features_db,
)
from labm8.py import app

FLAGS = app.FLAGS
app.DEFINE_string(
  "db",
  "sqlite:///tmp/phd/experimental/deplearning/clgen/closeness_to_grewe_features/db.db",
  "URL of the database to import OpenCL kernels to.",
)


def RowToStaticFeatures(
  row: typing.Dict[str, typing.Any], datafolder
) -> grewe_features_db.StaticFeatures:
  path = ncc.DataFrameRowToKernelSrcPath(row, datafolder)
  with open(path, "rb") as f:
    src = f.read().decode("unicode_escape")
  src = src.encode("ascii", "ignore").decode("ascii")

  identifier = ":".join(
    [
      row["program:benchmark_suite_name"],
      row["program:benchmark_name"],
      row["program:opencl_kernel_name"],
    ]
  )

  return grewe_features_db.StaticFeatures(
    src_sha256=hashlib.sha256(src.encode("utf-8")).hexdigest(),
    origin=f"benchmarks_{identifier}",
    grewe_compute_operation_count=row["feature:comp"],
    grewe_rational_operation_count=row["feature:rational"],
    grewe_global_memory_access_count=row["feature:mem"],
    grewe_local_memory_access_count=row["feature:localmem"],
    grewe_coalesced_memory_access_count=row["feature:coalesced"],
    grewe_atomic_operation_count=row["feature:atomic"],
    src=src,
  )


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))

  db = grewe_features_db.Database(FLAGS.db)
  df = opencl_device_mapping_dataset.OpenClDeviceMappingsDataset().df

  new_count = 0
  with ncc.DEEPTUNE_INST2VEC_DATA_ARCHIVE as datafolder:
    for _, row in progressbar.progressbar(list(df.iterrows())):
      with db.Session(commit=True) as session:
        obj = RowToStaticFeatures(row, datafolder)
        # Check if it already exists in the database.
        exists = (
          session.query(grewe_features_db.StaticFeatures)
          .filter_by(src_sha256=obj.src_sha256)
          .filter(grewe_features_db.StaticFeatures.origin.like("benchmarks_%"))
          .first()
        )
        if not exists:
          new_count += 1
          session.add(obj)

  app.Log(1, "Added %d new database entries", new_count)


if __name__ == "__main__":
  app.RunWithArgs(main)
