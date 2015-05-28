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

import labm8 as lab
import labm8.fs
from labm8 import system

import os
import re

class TestFs(TestCase):

    # path()
    def test_path(self):
        self._test("foo/bar", lab.fs.path("foo", "bar"))
        self._test("foo/bar/car", lab.fs.path("foo/bar", "car"))

    def test_path_homedir(self):
        self._test("/home",
                   re.search("^(/home).*", lab.fs.path("~", "foo")).group(1))


    # abspath()
    def test_abspath(self):
        self._test(os.path.abspath(".") + "/foo/bar",
                   lab.fs.abspath("foo", "bar"))
        self._test(os.path.abspath(".") + "/foo/bar/car",
                   lab.fs.abspath("foo/bar", "car"))

    def test_abspath_homedir(self):
        self._test("/home",
                   re.search("^(/home).*", lab.fs.abspath("~", "foo")).group(1))


    # is_subdir()
    def test_is_subdir(self):
        self._test(True, lab.fs.is_subdir("/home", "/"))
        self._test(True, lab.fs.is_subdir("/proc/1", "/proc"))
        self._test(True, lab.fs.is_subdir("/proc/1", "/proc/1/"))

    def test_is_subdir_not_subdir(self):
        self._test(False,
                   lab.fs.is_subdir("/", "/home"))


    # basename()
    def test_basename(self):
        self._test("foo", lab.fs.basename("foo"))
        self._test("foo", lab.fs.basename(lab.fs.abspath("foo")))


    # cd(), cdpop()
    def test_cd(self):
        cwd = os.getcwd()

        print("CWD", cwd)
        lab.fs.cd("..")
        print("CWD", os.getcwd())
        cwd = lab.fs.cdpop()
        print("CWD", cwd)
        #self._test(cwd, lab.fs.cdpop())


    # cdpop()
    def test_cdpop(self):
        cwd = os.getcwd()
        for i in range(10):
            self._test(cwd, lab.fs.cdpop())


    # pwd()
    def test_pwd(self):
        self._test(os.getcwd(), lab.fs.pwd())


    # exists()
    def test_exists(self):
        self._test(True, lab.fs.exists(__file__))
        self._test(True, lab.fs.exists("/"))
        self._test(False, lab.fs.exists("/not/a/real/path (I hope!)"))


    # isfile()
    def test_isfile(self):
        self._test(True, lab.fs.isfile(__file__))
        self._test(False, lab.fs.isfile("/"))
        self._test(False, lab.fs.isfile("/not/a/real/path (I hope!)"))


    # isdir()
    def test_isdir(self):
        self._test(False, lab.fs.isdir(__file__))
        self._test(True, lab.fs.isdir("/"))
        self._test(False, lab.fs.isdir("/not/a/real/path (I hope!)"))


    # notified_watchers()
    def test_notified_watchers_empty(self):
        self._test(set(), lab.fs.notified_watchers("/home"))
        self._test(set(), lab.fs.notified_watchers("/"))

    def test_notified_watchers(self):
        home = lab.fs.Watcher("/home")
        root = lab.fs.Watcher("/")
        watchers = set([home, root])

        # Register watchers.
        for watcher in watchers:
            lab.fs.register(watcher)

        self._test(watchers, lab.fs.notified_watchers("/home"))
        self._test(watchers, lab.fs.notified_watchers("/home/foo"))
        self._test(set([root]), lab.fs.notified_watchers("/tmp"))

        # Test teardown.
        for watcher in watchers:
            lab.fs.unregister(watcher)

    # read()
    def test_read_empty(self):
        self._test([],
                   lab.fs.read("tests/data/empty_file"))

    def test_read_hello_world(self):
        self._test(['Hello, world!'],
                   lab.fs.read("tests/data/hello_world"))

    def test_read_data1(self):
        print("PWD", os.getcwd())
        self._test([
            '# data1 - test file',
            'This',
            'is a test file',
            'With',
            'trailing  # comment',
            '',
            '',
            '',
            'whitespace',
            '0.344'
        ],
                   lab.fs.read("tests/data/data1"))

    def test_read_data1_comment(self):
        self._test([
            'This',
            'is a test file',
            'With',
            'trailing',
            '',
            '',
            '',
            'whitespace',
            '0.344'
        ],
                   lab.fs.read("tests/data/data1", comment_char="#"))

    def test_read_data1_no_rstrip(self):
        self._test([
            '# data1 - test file\n',
            'This\n',
            'is a test file\n',
            'With\n',
            'trailing  # comment  \n',
            '\n',
            '\n',
            '\n',
            'whitespace\n',
            '0.344\n'
        ],
                   lab.fs.read("tests/data/data1", rstrip=False))

    # rm()
    def test_rm(self):
        system.echo("Hello, world!", "/tmp/labm8.tmp")
        self._test(True, lab.fs.isfile("/tmp/labm8.tmp"))
        lab.fs.rm("/tmp/labm8.tmp")
        self._test(False, lab.fs.isfile("/tmp/labm8.tmp"))
        lab.fs.rm("/tmp/labm8.tmp")
        lab.fs.rm("/tmp/labm8.tmp")


if __name__ == '__main__':
    main()
