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
from unittest import TestCase, main

import labm8 as lab
import labm8.crypto

class TestCrypto(TestCase):

    # sha1()
    def test_sha1_empty_str(self):
        self.assertTrue(lab.crypto.sha1("")
                        == "da39a3ee5e6b4b0d3255bfef95601890afd80709")

    def test_sha1_hello_world(self):
        self.assertTrue(lab.crypto.sha1("Hello, World!")
                        == "0a0a9f2a6772942557ab5355d76af442f8f65e01")

    # sha1_file()
    def test_sha1_file_empty(self):
        sha1 = lab.crypto.sha1_file("tests/data/empty_file")
        self.assertTrue(sha1 == "da39a3ee5e6b4b0d3255bfef95601890afd80709")

    def test_sha1_file_hello_world(self):
        sha1 = lab.crypto.sha1_file("tests/data/hello_world")
        self.assertTrue(sha1 == "09fac8dbfd27bd9b4d23a00eb648aa751789536d")

if __name__ == '__main__':
    main()
