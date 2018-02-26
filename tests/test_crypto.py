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

import labm8 as lab

class TestCrypto(TestCase):

    # sha1()
    def test_sha1_empty_str(self):
        self._test("da39a3ee5e6b4b0d3255bfef95601890afd80709",
                   lab.crypto.sha1_str(""))

    def test_sha1_hello_world(self):
        self._test("0a0a9f2a6772942557ab5355d76af442f8f65e01",
                   lab.crypto.sha1_str("Hello, World!"))

    # sha1_list()
    def test_sha1_empty_list(self):
        self._test("da39a3ee5e6b4b0d3255bfef95601890afd80709",
                   lab.crypto.sha1_list())
        self._test("97d170e1550eee4afc0af065b78cda302a97674c",
                   lab.crypto.sha1_list([]))

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

    # md5()
    def test_md5_empty_str(self):
        self._test("d41d8cd98f00b204e9800998ecf8427e",
                   lab.crypto.md5_str(""))

    def test_md5_hello_world(self):
        self._test("65a8e27d8879283831b664bd8b7f0ad4",
                   lab.crypto.md5_str("Hello, World!"))

    # md5_list()
    def test_md5_empty_list(self):
        self._test("d41d8cd98f00b204e9800998ecf8427e",
                   lab.crypto.md5_list())
        self._test("d751713988987e9331980363e24189ce",
                   lab.crypto.md5_list([]))

    def test_md5_list(self):
        self._test("6ded24f0b2f43dd31e601a27fcecb7e8",
                   lab.crypto.md5_list(["hello", "world"]))

    # md5_file()
    def test_md5_file_empty(self):
        self._test("d41d8cd98f00b204e9800998ecf8427e",
                   lab.crypto.md5_file("tests/data/empty_file"))

    def test_md5_file_hello_world(self):
        self._test("746308829575e17c3331bbcb00c0898b",
                   lab.crypto.md5_file("tests/data/hello_world"))

    # sha256()
    def test_sha256_empty_str(self):
        self._test("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                   lab.crypto.sha256_str(""))

    def test_sha256_hello_world(self):
        self._test("dffd6021bb2bd5b0af676290809ec3a53191dd81c7f70a4b28688a362182986f",
                   lab.crypto.sha256_str("Hello, World!"))

    # sha256_list()
    def test_sha256_empty_list(self):
        self._test("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                   lab.crypto.sha256_list())
        self._test("4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
                   lab.crypto.sha256_list([]))

    def test_sha256_list(self):
        self._test("be3d036085587af9522a8358dd1d09ba2b0ec63db92a62d28cf00dfcaeb25ca1",
                   lab.crypto.sha256_list(["hello", "world"]))

    # sha256_file()
    def test_sha256_file_empty(self):
        self._test("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                   lab.crypto.sha256_file("tests/data/empty_file"))

    def test_sha256_file_hello_world(self):
        self._test("d9014c4624844aa5bac314773d6b689ad467fa4e1d1a50a1b8a99d5a95f72ff5",
                   lab.crypto.sha256_file("tests/data/hello_world"))
