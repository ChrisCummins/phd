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

import labm8 as lab
import labm8.crypto

class TestCrypto(TestCase):

    # sha1()
    def test_sha1_empty_str(self):
        self._test("da39a3ee5e6b4b0d3255bfef95601890afd80709",
                   lab.crypto.sha1_str(""))
        self._test("da39a3ee5e6b4b0d3255bfef95601890afd80709",
                   lab.crypto.sha1_str("".decode("utf-8")))

    def test_sha1_hello_world(self):
        self._test("0a0a9f2a6772942557ab5355d76af442f8f65e01",
                   lab.crypto.sha1_str("Hello, World!"))

    # sha1_list()
    def test_sha1_list(self):
        self._test("06bf71070d31b2ebe4bdae828fc76a70e4b56f00",
                   lab.crypto.sha1_list(["hello", "world"]))

    # sha1_file()
    def test_sha1_file_empty(self):
        self._test("da39a3ee5e6b4b0d3255bfef95601890afd80709",
                   lab.crypto.sha1_file("tests/data/empty_file"))

    def test_sha1_file_hello_world(self):
        self._test("09fac8dbfd27bd9b4d23a00eb648aa751789536d",
                   lab.crypto.sha1_file("tests/data/hello_world"))
