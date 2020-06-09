# Copyright (c) 2016-2020 Chris Cummins.
#
# clgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# clgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with clgen.  If not, see <https://www.gnu.org/licenses/>.
"""Export the pre-processed contentfiles from a database."""
from hashlib import sha256
from pathlib import Path

import sqlalchemy as sql
from tqdm import tqdm

from deeplearning.clgen.corpuses.preprocessed import PreprocessedContentFile
from deeplearning.clgen.corpuses.preprocessed import PreprocessedContentFiles
from labm8.py import app
from labm8.py.sqlutil import OffsetLimitBatchedQuery

FLAGS = app.FLAGS

app.DEFINE_string(
  "db", None, "Path of a SQLite database of pre-processed contentfiles."
)
app.DEFINE_output_path(
  "path", "/tmp/phd/preprocessed", "The directory to export files to.",
)
app.DEFINE_string("suffix", ".txt", "The filename suffix for exported files.")


def Main():
  """Main entry point."""
  db = PreprocessedContentFiles(FLAGS.db, must_exist=True)
  outpath: Path = FLAGS.path
  outpath.mkdir(parents=True, exist_ok=True)

  i = 0
  with db.Session() as session:
    query = session.query(PreprocessedContentFile.text).filter(
      PreprocessedContentFile.preprocessing_succeeded == True
    )
    n = (
      session.query(sql.func.count(PreprocessedContentFile.id)).filter(
        PreprocessedContentFile.preprocessing_succeeded == True
      )
    ).scalar()

    for batch in OffsetLimitBatchedQuery(query, batch_size=16384):
      for row in tqdm(batch.rows, initial=i, total=n):
        text = row[0]
        name = f"{sha256(text.encode('utf-8')).hexdigest()}{FLAGS.suffix}"
        with open(outpath / name, "w") as f:
          f.write(text.encode("ascii", "ignore").decode("ascii"))
      i += len(batch.rows)


if __name__ == "__main__":
  app.Run(Main)
