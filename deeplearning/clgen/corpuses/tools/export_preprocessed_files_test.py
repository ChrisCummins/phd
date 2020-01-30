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
"""Unit tests for //deeplearning/clgen/corpuses:preprocessed."""
import pathlib

from deeplearning.clgen.corpuses import preprocessed
from deeplearning.clgen.corpuses.tools import export_preprocessed_files
from labm8.py import app
from labm8.py import fs
from labm8.py import test

FLAGS = test.FLAGS


@test.Fixture(scope="function")
def preprocessed_db(
  tempdir: pathlib.Path,
) -> preprocessed.PreprocessedContentFiles:
  """A preprocessed database with three files:

    a -> Hello, world
    a2 -> This is a duplicate (has same sha256 as 'a')
    b -> Hello, foo
    c -> ERROR: failure (not successfully preprocessed)
  """
  db = preprocessed.PreprocessedContentFiles(
    f"sqlite:///{tempdir}/preprocessed.db"
  )

  with db.Session(commit=True) as session:
    session.add_all(
      [
        preprocessed.PreprocessedContentFile(
          input_relpath="a",
          input_sha256="00000000",
          input_charcount=10,
          input_linecount=10,
          sha256="00000000",
          charcount=10,
          linecount=1,
          text="Hello, world",
          preprocessing_succeeded=True,
          preprocess_time_ms=4,
          wall_time_ms=4,
        ),
        preprocessed.PreprocessedContentFile(
          input_relpath="a2",
          input_sha256="00000000",
          input_charcount=10,
          input_linecount=10,
          sha256="00000000",
          charcount=10,
          linecount=1,
          text="This is a duplicate",
          preprocessing_succeeded=True,
          preprocess_time_ms=4,
          wall_time_ms=4,
        ),
        preprocessed.PreprocessedContentFile(
          input_relpath="b",
          input_sha256="11111111",
          input_charcount=10,
          input_linecount=10,
          sha256="11111111",
          charcount=10,
          linecount=1,
          text="Hello, foo",
          preprocessing_succeeded=True,
          preprocess_time_ms=4,
          wall_time_ms=4,
        ),
        preprocessed.PreprocessedContentFile(
          input_relpath="c",
          input_sha256="22222222",
          input_charcount=10,
          input_linecount=10,
          sha256="22222222",
          charcount=10,
          linecount=1,
          text="ERROR: failure",
          preprocessing_succeeded=False,
          preprocess_time_ms=4,
          wall_time_ms=4,
        ),
      ]
    )
  yield db


def test_ExportFiles(
  preprocessed_db: preprocessed.PreprocessedContentFiles, tempdir: pathlib.Path
):
  export_preprocessed_files.ExportPreprocessedFiles(
    preprocessed_db, tempdir / "out"
  )
  assert fs.Read(tempdir / "out" / "00000000.txt") == "Hello, world"
  assert fs.Read(tempdir / "out" / "11111111.txt") == "Hello, foo"
  assert fs.Read(tempdir / "out" / "22222222.txt") == "ERROR: failure"
  assert len(list((tempdir / "out").iterdir())) == 3


def test_ExportFiles_only_succesfully_preprocessed(
  preprocessed_db: preprocessed.PreprocessedContentFiles, tempdir: pathlib.Path
):
  export_preprocessed_files.ExportPreprocessedFiles(
    preprocessed_db, tempdir / "out", only_successfully_preprocessed=True
  )
  assert fs.Read(tempdir / "out" / "00000000.txt") == "Hello, world"
  assert fs.Read(tempdir / "out" / "11111111.txt") == "Hello, foo"
  assert not (tempdir / "out" / "22222222.txt").is_file()
  assert len(list((tempdir / "out").iterdir())) == 2


def test_ExportFiles_with_file_suffix(
  preprocessed_db: preprocessed.PreprocessedContentFiles, tempdir: pathlib.Path
):
  export_preprocessed_files.ExportPreprocessedFiles(
    preprocessed_db, tempdir / "out", file_suffix=".cl"
  )
  assert fs.Read(tempdir / "out" / "00000000.cl") == "Hello, world"
  assert fs.Read(tempdir / "out" / "11111111.cl") == "Hello, foo"
  assert fs.Read(tempdir / "out" / "22222222.cl") == "ERROR: failure"
  assert len(list((tempdir / "out").iterdir())) == 3


def test_Main(
  preprocessed_db: preprocessed.PreprocessedContentFiles, tempdir: pathlib.Path
):
  FLAGS.unparse_flags()
  FLAGS(
    [
      "argv0",
      "--db",
      preprocessed_db.url,
      "--outdir",
      str(tempdir / "out"),
      "--only_successfully_preprocessed",
    ]
  )
  export_preprocessed_files.Main()
  assert fs.Read(tempdir / "out" / "00000000.txt") == "Hello, world"
  assert fs.Read(tempdir / "out" / "11111111.txt") == "Hello, foo"
  assert not (tempdir / "out" / "22222222.txt").is_file()
  assert len(list((tempdir / "out").iterdir())) == 2


def test_Main_missing_required_args():
  FLAGS.unparse_flags()
  FLAGS(["argv0"])
  with test.Raises(app.UsageError):
    export_preprocessed_files.Main()


if __name__ == "__main__":
  test.Main()
