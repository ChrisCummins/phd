"""Import features from the 1000 CLgen kernels used in CGO'17 paper."""
import typing

from experimental.deeplearning.clgen.closeness_to_grewe_features import (
  grewe_features_db,
)
from labm8.py import app
from labm8.py import bazelutil

FLAGS = app.FLAGS
app.DEFINE_string(
  "db",
  "sqlite:///tmp/phd/experimental/deplearning/clgen/closeness_to_grewe_features/db.db",
  "URL of the database to import OpenCL kernels to.",
)

_CLGEN_1000_TAR = bazelutil.DataArchive(
  "phd/docs/2017_02_cgo/data/clgen-1000.tar.bz2"
)


def main(argv: typing.List[str]):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(" ".join(argv[1:])))

  db = grewe_features_db.Database(FLAGS.db)

  with _CLGEN_1000_TAR as kernels_dir:
    paths_to_import = list((kernels_dir / "clgen-1000/kernels").iterdir())
    assert len(paths_to_import) == 1000
    db.ImportStaticFeaturesFromPaths(paths_to_import, "clgen_1000")

  app.Log(1, "done")


if __name__ == "__main__":
  app.RunWithArgs(main)
