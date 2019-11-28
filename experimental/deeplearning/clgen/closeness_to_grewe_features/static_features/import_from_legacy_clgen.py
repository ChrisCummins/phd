"""Import from legacy CLgen sampler database format."""
import math
import multiprocessing
import pathlib
import sqlite3
import tempfile
import typing

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
app.DEFINE_string(
  "legacy_clgen_db", None, "Path of the legacy CLgen sqlite database."
)
app.DEFINE_string(
  "origin", "clgen_legacy", 'Name of the origin of the kernels, e.g. "github".'
)
app.DEFINE_integer(
  "batch_size", 256, "The number of kernels to process in a batch."
)
app.DEFINE_integer(
  "grewe_db_import_process_count",
  multiprocessing.cpu_count(),
  "The number of processes to spawn when importing files to database.",
)


def BatchQueryResults(query):
  """Batch results of a query."""
  i = 0
  batch = []
  for row in query:
    batch.append(row)
    i += 1
    if i >= FLAGS.batch_size:
      yield batch
      batch = []
      i = 0


def CreateTempFileFromTestcase(
  tempdir: pathlib.Path, src: str, number: int
) -> pathlib.Path:
  """Write testcase to a file in directory."""
  path = tempdir / f"{number}.cl"
  with open(path, "w") as f:
    f.write(src)
  return path


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))

  conn = sqlite3.connect(FLAGS.legacy_clgen_db)
  c = conn.cursor()

  (num_kernels,) = c.execute(
    "SELECT COUNT(*) FROM PreprocessedKernels"
  ).fetchone()
  app.Log(1, "Database contains %d kernels", num_kernels)
  num_batches = math.ceil(num_kernels / FLAGS.batch_size)

  batches = BatchQueryResults(
    c.execute("SELECT kernel FROM PreprocessedKernels")
  )

  prefix = "phd_experimental_deeplearning_clgen_"
  with multiprocessing.Pool() as pool:
    for i, batch in enumerate(batches):
      with tempfile.TemporaryDirectory(prefix=prefix) as d:
        app.Log(1, "Batch %d of %d", i + 1, num_batches)
        d = pathlib.Path(d)
        paths_to_import = [
          CreateTempFileFromTestcase(d, src, i)
          for i, (src,) in enumerate(batch)
        ]
        db = grewe_features_db.Database(FLAGS.db)
        success_count, new_row_count = db.ImportStaticFeaturesFromPaths(
          paths_to_import, FLAGS.origin, pool
        )
        app.Log(
          1,
          "Extracted features from %d of %d kernels, %d new rows",
          success_count,
          FLAGS.batch_size,
          new_row_count,
        )


if __name__ == "__main__":
  app.RunWithArgs(main)
