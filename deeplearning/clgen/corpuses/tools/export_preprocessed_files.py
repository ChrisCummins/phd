"""Export the preprocessed text of a database to files or compressed archives.

"""
import io
import pathlib
import tarfile
from typing import Optional

import sqlalchemy as sql
from labm8.py import app
from labm8.py import fs
from labm8.py import humanize
from labm8.py import ppar
from labm8.py import sqlutil

from deeplearning.clgen.corpuses import preprocessed

FLAGS = app.FLAGS

app.DEFINE_database(
  "db",
  preprocessed.PreprocessedContentFiles,
  None,
  "The URL of the database to export files from.",
)
app.DEFINE_output_path(
  "outdir", None, "The output directory to export files to."
)
app.DEFINE_string(
  "file_suffix", ".txt", "The filename suffix for exported texts."
)
app.DEFINE_boolean(
  "only_successfully_preprocessed",
  False,
  "Export only the text of contenfiles which were succesfully preprocessed.",
)
app.DEFINE_integer(
  "batch_size", 10000, "The number of texts to export in a batch."
)
app.DEFINE_boolean(
  "archive_batches",
  False,
  "Create a .tar.bz2 archive of each --batch_size files.",
)


def ExportPreprocessedFiles(
  db: preprocessed.PreprocessedContentFiles,
  outdir: pathlib.Path,
  only_successfully_preprocessed: bool = False,
  file_suffix: str = ".txt",
  batch_size: int = 10000,
  archive_batches: bool = False,
):
  """Export the texts from a database of preprocessed contenfiles.

  This creates a file for each text in the output directory, whose filename is
  the sha256 of the pre-processed text.

  Args:
    db: The database to export texts from.
    outdir: The directory to write texts to.
    only_successfully_preprocessed: Filter only the pre-processed texts that
      were succesfully preprocessed.
    file_suffix: The filename suffix for exported texts.
    batch_size: The size of batches of texts to export.
    archive_batches: Create a .tar.bz2 archive of each batch of files.
  """
  outdir.mkdir(parents=True, exist_ok=True)

  # Compute the number of files there are to export.
  with db.Session() as session:
    max_rows = session.query(
      sql.func.count(
        sql.func.distinct(preprocessed.PreprocessedContentFile.sha256)
      )
    ).scalar()
    app.Log(1, "%s files to export", humanize.Commas(max_rows))

  # Create a new session because we are going to hand over the session object
  # to a background thread, which is not supported in SQLite.
  with db.Session() as session:
    # Get the IDs of the unique sha256 texts.
    ids_to_export = session.query(
      sql.func.min(preprocessed.PreprocessedContentFile.id)
    ).group_by(preprocessed.PreprocessedContentFile.sha256)
    if only_successfully_preprocessed:
      ids_to_export = ids_to_export.filter(
        preprocessed.PreprocessedContentFile.preprocessing_succeeded == True
      )

    # Get the sha256s and texts to export.
    query = session.query(
      preprocessed.PreprocessedContentFile.sha256,
      preprocessed.PreprocessedContentFile.text,
    ).filter(preprocessed.PreprocessedContentFile.id.in_(ids_to_export))

    # Read batches of this query in a parallel thread.
    # query_batches = sqlutil.OffsetLimitBatchedQuery(query, batch_size=batch_size)
    query_batches = ppar.ThreadedIterator(
      sqlutil.OffsetLimitBatchedQuery(query, batch_size=batch_size),
      max_queue_size=5,
    )

    for i, batch in enumerate(query_batches, start=1):
      if archive_batches:
        with tarfile.open(outdir / f"preprocessed_{i}.tar.bz2", "w:bz2") as tar:
          for sha256, text in batch.rows:
            info = tarfile.TarInfo(name=f"{sha256}{file_suffix}")
            info.size = len(text)
            tar.addfile(info, io.BytesIO(text.encode("utf-8")))
      else:
        for sha256, text in batch.rows:
          fs.Write(outdir / f"{sha256}{file_suffix}", text.encode("utf-8"))

      app.Log(
        1,
        "Exported pre-processed files %s..%s of %s (%.2f%%)",
        humanize.Commas(batch.offset),
        humanize.Commas(batch.offset + len(batch.rows)),
        humanize.Commas(max_rows),
        ((batch.offset + len(batch.rows)) / max_rows) * 100,
      )


def Main():
  """Main entry point."""
  if not FLAGS.db:
    raise app.UsageError("Flag is required: --db")
  if not FLAGS.outdir:
    raise app.UsageError("Flag is required: --outdir")

  ExportPreprocessedFiles(
    FLAGS.db(),
    FLAGS.outdir,
    file_suffix=FLAGS.file_suffix,
    only_successfully_preprocessed=FLAGS.only_successfully_preprocessed,
    batch_size=FLAGS.batch_size,
    archive_batches=FLAGS.archive_batches,
  )


if __name__ == "__main__":
  app.Run(Main)
