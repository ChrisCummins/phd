from __future__ import print_function

from unittest import TestCase
import tests

import os
import sqlite3
import sys

import labm8
from labm8 import fs

import smith
from smith import dbutil
from smith import preprocess
from smith import config

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

    def test_strip_attributes(self):
        self.assertEqual("", preprocess.strip_attributes(
            "__attribute__((reqd_work_group_size(64,1,1)))"))

        out = "foobar"
        tin = "foo__attribute__((reqd_work_group_size(WG_SIZE,1,1)))bar"
        self.assertEqual(out, preprocess.strip_attributes(tin))

        out = "typedef  unsigned char uchar8;"
        tin = "typedef __attribute__((ext_vector_type(8))) unsigned char uchar8;"
        self.assertEqual(out, preprocess.strip_attributes(tin))

        out = ("typedef  unsigned char uchar8;\n"
               "typedef  unsigned char uchar8;")
        tin = ("typedef __attribute__  ((ext_vector_type(8))) unsigned char uchar8;\n"
               "typedef __attribute__((reqd_work_group_size(64,1,1))) unsigned char uchar8;")
        self.assertEqual(out, preprocess.strip_attributes(tin))

    def test_vacuum(self):
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

        preprocess.vacuum("tmp.db")

        # Check that vacuum worked:
        db = sqlite3.connect("tmp.db")
        c = db.cursor()
        c.execute("SELECT Count(*) FROM PreprocessedFiles")
        count = c.fetchone()[0]
        self.assertEqual(3, count)
        c.execute("SELECT contents FROM PreprocessedFiles WHERE status=1 OR status=2")
        rows = c.fetchall()
        print(rows)
        self.assertTrue(all(not r == "[VACUUMED]" for r in rows))

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


if __name__ == '__main__':
    main()
