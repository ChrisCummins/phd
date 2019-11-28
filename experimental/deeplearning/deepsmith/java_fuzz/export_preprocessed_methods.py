"""Export pre-processed methods.

This is a utility script for dumping methods to inspect / test with.
"""
import pathlib

import progressbar

from deeplearning.clgen.corpuses import preprocessed
from labm8.py import app
from labm8.py import fs

FLAGS = app.FLAGS
app.DEFINE_database(
  "db",
  preprocessed.PreprocessedContentFiles,
  "sqlite:////var/phd/experimental/deeplearning/deepsmith/java_fuzz/preprocessed.db",
  "URL of the database of preprocessed Java methods.",
)
app.DEFINE_output_path(
  "outdir",
  "/tmp/phd/experimental/deeplearning/deepsmith/java_fuzz/preprocessed",
  "Directory to write preprocessed methods to.",
  is_dir=True,
)
app.DEFINE_integer("n", 10000, "The number of methods to export.")


def ExportPreprocessed(
  db: preprocessed.PreprocessedContentFiles, outdir: pathlib.Path, n: int
):
  outdir.mkdir(parents=True, exist_ok=True)

  with db.Session() as s:
    q = (
      s.query(
        preprocessed.PreprocessedContentFile.sha256,
        preprocessed.PreprocessedContentFile.text,
      )
      .filter(
        preprocessed.PreprocessedContentFile.preprocessing_succeeded == True
      )
      .limit(n)
    )

    for i, (sha256, text) in enumerate(progressbar.ProgressBar()(q)):
      fs.Write(outdir / f"{sha256}.txt", text.encode("utf-8"))


def main():
  """Main entry point."""
  ExportPreprocessed(FLAGS.db(), FLAGS.outdir, FLAGS.n)


if __name__ == "__main__":
  app.Run(main)
