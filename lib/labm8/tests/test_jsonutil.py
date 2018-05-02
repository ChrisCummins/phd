# Copyright (C) 2015-2017 Chris Cummins.
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

from labm8 import fs
from labm8 import system
from labm8 import jsonutil


class TestJsonutil(TestCase):
    def test_loads(self):
        a_str = """{
            "a": 1,  // this has comments
            "b": [1, 2, 3]
        } # end comment
        // begin with comment
        """
        a = jsonutil.loads(a_str)

        self.assertEqual(a["a"], 1)
        self.assertEqual(a["b"], [1, 2, 3])
        self.assertFalse("c" in a)

    def test_loads_malformed(self):
        a_str = """bad json {asd,,}"""
        with self.assertRaises(ValueError):
            jsonutil.loads(a_str)

    def test_read_file(self):
        a_str = """{
            "a": 1,  // this has comments
            "b": [1, 2, 3]
        } # end comment
        // begin with comment
        """
        system.echo(a_str, "/tmp/labm8.loaf.json")
        a = jsonutil.read_file("/tmp/labm8.loaf.json")

        self.assertEqual(a["a"], 1)
        self.assertEqual(a["b"], [1, 2, 3])
        self.assertFalse("c" in a)

    def test_read_file_bad_path(self):
        with self.assertRaises(fs.File404):
            jsonutil.read_file("/not/a/real/path")

        self.assertEqual({}, jsonutil.read_file("/not/a/real/path",
                                                must_exist=False))

    def test_write_file(self):
        d1 = {
            "a": "1",
            "b": "2"
        }
        jsonutil.write_file("/tmp/labm8.write_file.json", d1)
        d2 = jsonutil.read_file("/tmp/labm8.write_file.json")
        fs.rm("/tmp/labm8.write_file.json")

        jsonutil.write_file("/tmp/labm8.write_file2.json", d1)
        d3 = jsonutil.read_file("/tmp/labm8.write_file2.json")
        fs.rm("/tmp/labm8.write_file2.json")

        self.assertEqual(d1, d2)
        self.assertEqual(d1, d3)
