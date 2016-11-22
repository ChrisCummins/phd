# Copyright (C) 2015, 2016 Chris Cummins.
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

import json
import json.decoder

import labm8 as lab
from labm8 import fs
from labm8 import system
from labm8 import jsonutil


class TestJsonutil(TestCase):
    def test_loads(self):
        a_str = """{
            "a": 1,  // this has comments
            "b": [1, 2, 3]
        }"""
        a = jsonutil.loads(a_str)

        self.assertEqual(a["a"], 1)
        self.assertEqual(a["b"], [1, 2, 3])
        self.assertFalse("c" in a)

    def test_loads_malformed(self):
        a_str = """bad json {asd,,}"""
        with self.assertRaises(ValueError):
            jsonutil.loads(a_str)

    def test_loadf(self):
        a_str = """{
            "a": 1,  // this has comments
            "b": [1, 2, 3]
        }"""
        system.echo(a_str, "/tmp/labm8.loaf.json")
        a = jsonutil.loadf("/tmp/labm8.loaf.json")

        self.assertEqual(a["a"], 1)
        self.assertEqual(a["b"], [1, 2, 3])
        self.assertFalse("c" in a)

    def test_loadf_bad_path(self):
        with self.assertRaises(fs.File404):
            jsonutil.loadf("/not/a/real/path")
