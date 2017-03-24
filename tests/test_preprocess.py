#
# Copyright 2016, 2017 Chris Cummins <chrisc.101@gmail.com>.
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
from unittest import TestCase
import tests

import os
import sqlite3
import sys

import labm8
from labm8 import fs

import clgen
from clgen import dbutil
from clgen import preprocess

# Invoke tests with UPDATE_GS_FILES set to update the gold standard
# tests. E.g.:
#
#   $ UPDATE_GS_FILES=1 python3 ./setup.py test
#
UPDATE_GS_FILES = True if 'UPDATE_GS_FILES' in os.environ else False

def preprocess_pair(basename, preprocessor=preprocess.preprocess):
    gs_path = tests.data_path(os.path.join('cl', str(basename) + '.gs'),
                              exists=not UPDATE_GS_FILES)
    tin_path = tests.data_path(os.path.join('cl', str(basename) + '.cl'))

    # Run preprocess
    tin = tests.data_str(tin_path)
    tout = preprocessor(tin)

    if UPDATE_GS_FILES:
        gs = tout
        with open(gs_path, 'w') as outfile:
            outfile.write(gs)
            print("\n-> updated gold standard file '{}' ..."
                  .format(gs_path), file=sys.stderr, end=' ')
    else:
        gs = tests.data_str(gs_path)

    return (gs, tout)


class TestPreprocess(TestCase):
    def test_preprocess(self):
        self.assertEqual(*preprocess_pair('sample-1'))

    def test_preprocess_shim(self):
        # FLOAT_T is defined in shim header
        self.assertTrue(preprocess.preprocess("""
__kernel void A(__global FLOAT_T* a) { int b; }""", use_shim=True))

        # Preprocess will fail without FLOAT_T defined
        with self.assertRaises(preprocess.BadCodeException):
            preprocess.preprocess("""
__kernel void A(__global FLOAT_T* a) { int b; }""", use_shim=False)

    def test_ugly_preprocessed(self):
        # empty kernel protoype is rejected
        with self.assertRaises(preprocess.NoCodeException):
            preprocess.preprocess("""\
__kernel void A() {
}\
""")
        # kernel containing some code returns the same.
        self.assertEqual("""\
__kernel void A() {
  int a;
}\
""", preprocess.preprocess("""\
__kernel void A() {
  int a;
}\
"""))

    def test_preprocess_stable(self):
        code = """\
__kernel void A(__global float* a) {
  int b;
  float c;
  int d = get_global_id(0);

  a[d] *= 2.0f;
}"""
        # pre-processing is "stable" if the code doesn't change
        out = code
        for _ in range(5):
            out = preprocess.preprocess(out)
            self.assertEqual(out, code)

    def test_remove_bad_preprocessed(self):
        fs.rm("tmp.db")
        dbutil.create_db("tmp.db")
        db = sqlite3.connect("tmp.db")
        c = db.cursor()

        # Create some data to test with:
        c.execute("DELETE FROM PreprocessedFiles")
        c.execute("INSERT INTO PreprocessedFiles VALUES(?,?,?)",
                  ("id1", 0, "good output"))
        c.execute("INSERT INTO PreprocessedFiles VALUES(?,?,?)",
                  ("id2", 1, "bad output"))
        c.execute("INSERT INTO PreprocessedFiles VALUES(?,?,?)",
                  ("id3", 2, "ugly output"))
        db.commit()
        c.close()

        # Check that data was written properly:
        c = db.cursor()
        c.execute("SELECT Count(*) FROM PreprocessedFiles")
        count = c.fetchone()[0]
        self.assertEqual(3, count)
        db.close()

        preprocess.remove_bad_preprocessed("tmp.db")

        # Check that clean worked:
        db = sqlite3.connect("tmp.db")
        c = db.cursor()
        c.execute("SELECT Count(*) FROM PreprocessedFiles")
        count = c.fetchone()[0]
        self.assertEqual(3, count)
        c.execute("SELECT contents FROM PreprocessedFiles WHERE status=1 "
                  "OR status=2")
        rows = c.fetchall()
        print(rows)
        self.assertTrue(all(not r == "[DELETED]" for r in rows))

        # Clean up:
        c.execute("DELETE FROM PreprocessedFiles")
        db.commit()
        c.close()

        # Check that clean-up worked:
        c = db.cursor()
        c.execute("SELECT Count(*) FROM PreprocessedFiles")
        count = c.fetchone()[0]
        self.assertEqual(0, count)
        fs.rm("tmp.db")
