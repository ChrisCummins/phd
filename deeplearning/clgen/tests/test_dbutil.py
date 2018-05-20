#
# Copyright 2016, 2017, 2018 Chris Cummins <chrisc.101@gmail.com>.
#
# This file is part of CLgen.
#
# CLgen is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CLgen is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CLgen.  If not, see <http://www.gnu.org/licenses/>.
#
import pytest

import deeplearning.tmp_clgen.clgen.errors
from deeplearning.tmp_clgen import dbutil
from deeplearning.tmp_clgen import test as tests
from lib.labm8 import fs


def test_create_db():
  db_path = tests.data_path("db", "tmp.db", exists=False)
  fs.rm(db_path)

  dbutil.create_db(db_path, github=False)
  assert fs.exists(db_path)

  with pytest.raises(deeplearning.tmp_clgen.clgen.errors.UserError):
    dbutil.create_db(db_path, github=False)


def test_create_db_gh():
  db_path = tests.data_path("db", "tmp.db", exists=False)
  fs.rm(db_path)

  dbutil.create_db(db_path, github=True)
  assert fs.exists(db_path)

  with pytest.raises(deeplearning.tmp_clgen.clgen.errors.UserError):
    dbutil.create_db(db_path, github=True)


def test_insert():
  db_path = tests.data_path("db", "tmp.db", exists=False)
  fs.rm(db_path)

  dbutil.create_db(db_path)
  db = dbutil.connect(db_path)
  c = db.cursor()

  assert dbutil.num_rows_in(db_path, "ContentFiles") == 0

  dbutil.sql_insert_dict(c, "ContentFiles", {"id": "a", "contents": "foo"})
  dbutil.sql_insert_dict(c, "PreprocessedFiles",
                         {"id": "a", "status": 0, "contents": "bar"})
  dbutil.sql_insert_dict(c, "PreprocessedFiles",
                         {"id": "b", "status": 1, "contents": "car"})

  db.commit()
  c = db.cursor()

  assert dbutil.num_rows_in(db_path, "ContentFiles") == 1
  assert dbutil.num_rows_in(db_path, "PreprocessedFiles") == 2

  assert dbutil.cc(db_path, "ContentFiles", "contents") == 3
  assert dbutil.cc(db_path, "ContentFiles", "id") == 1
  assert dbutil.lc(db_path, "ContentFiles", "contents") == 1

  dbutil.remove_bad_preprocessed(db_path)
  assert dbutil.num_rows_in(db_path, "ContentFiles") == 1
  # remove_bad_preprocessed doesn't actually delete any rows, just
  # replaces contents
  assert dbutil.num_rows_in(db_path, "PreprocessedFiles") == 2

  dbutil.remove_preprocessed(db_path)
  assert dbutil.num_rows_in(db_path, "ContentFiles") == 1
  assert dbutil.num_rows_in(db_path, "PreprocessedFiles") == 0
