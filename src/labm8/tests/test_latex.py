# Copyright (C) 2015 Chris Cummins.
#
# Labm8 is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# Labm8 is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with labm8.  If not, see <http://www.gnu.org/licenses/>.
from unittest import main
from tests import TestCase

import sqlite3 as sql

import pandas.io.sql as panda

import labm8 as lab
from labm8 import db
from labm8 import fs
from labm8 import latex

class TestLatex(TestCase):

    # escape()
    def test_escape(self):
        self._test("foo", latex.escape("foo"))
        self._test("123", latex.escape(123))
        self._test("foo\\_bar", latex.escape("foo_bar"))
        self._test("foo\\_bar\\_baz", latex.escape("foo_bar_baz"))
        self._test("foo\\_\\_bar", latex.escape("foo__bar"))

    # write_table_body()
    def test_write_table_body(self):
        self._test("1 & foo\\\\\n"
                   "2 & bar\\\\\n",
                   latex.write_table_body(((1, "foo"), (2, "bar"))))

    def test_write_table_body_headers(self):
        self._test("\\textbf{A} & \\textbf{B}\\\\\n"
                   "\\hline\n"
                   "1 & foo\\\\\n"
                   "2 & bar\\\\\n",
                   latex.write_table_body(((1, "foo"), (2, "bar")),
                                          headers=("A", "B")))

    def test_write_table_body_headers_no_hline(self):
        self._test("\\textbf{A} & \\textbf{B}\\\\\n"
                   "1 & foo\\\\\n"
                   "2 & bar\\\\\n",
                   latex.write_table_body(((1, "foo"), (2, "bar")),
                                          headers=("A", "B"),
                                          hline_after_header=False))

    def test_write_table_body_headers_no_fmt(self):
        self._test("A & B\\\\\n"
                   "\\hline\n"
                   "1 & foo\\\\\n"
                   "2 & bar\\\\\n",
                   latex.write_table_body(((1, "foo"), (2, "bar")),
                                          headers=("A", "B"),
                                          header_fmt=lambda x: x))


if __name__ == '__main__':
    main()
