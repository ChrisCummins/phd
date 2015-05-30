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

import os
import re

import labm8 as lab
from labm8 import fs
from labm8 import system


class TestFs(TestCase):

    # path()
    def test_path(self):
        self._test("foo/bar", fs.path("foo", "bar"))
        self._test("foo/bar/car", fs.path("foo/bar", "car"))

    def test_path_homedir(self):
        self._test("/home",
                   re.search("^(/home).*", fs.path("~", "foo")).group(1))


    # abspath()
    def test_abspath(self):
        self._test(os.path.abspath(".") + "/foo/bar",
                   fs.abspath("foo", "bar"))
        self._test(os.path.abspath(".") + "/foo/bar/car",
                   fs.abspath("foo/bar", "car"))

    def test_abspath_homedir(self):
        self._test("/home",
                   re.search("^(/home).*", fs.abspath("~", "foo")).group(1))


    # is_subdir()
    def test_is_subdir(self):
        self._test(True,  fs.is_subdir("/home", "/"))
        self._test(True,  fs.is_subdir("/proc/1", "/proc"))
        self._test(True,  fs.is_subdir("/proc/1", "/proc/1/"))
        self._test(False, fs.is_subdir("/proc/3", "/proc/1/"))
        self._test(False, fs.is_subdir("/", "/home"))

    def test_is_subdir_not_subdir(self):
        self._test(False,
                   fs.is_subdir("/", "/home"))


    # basename()
    def test_basename(self):
        self._test("foo", fs.basename("foo"))
        self._test("foo", fs.basename(fs.abspath("foo")))


    def test_dirname(self):
        self._test("", fs.dirname("foo"))
        self._test("/tmp", fs.dirname("/tmp/labm8.tmp"))


    # cd(), cdpop()
    def test_cd(self):
        cwd = os.getcwd()

        print("CWD", cwd)
        fs.cd("..")
        print("CWD", os.getcwd())
        cwd = fs.cdpop()
        print("CWD", cwd)
        #self._test(cwd, fs.cdpop())


    # cdpop()
    def test_cdpop(self):
        cwd = os.getcwd()
        for i in range(10):
            self._test(cwd, fs.cdpop())


    # pwd()
    def test_pwd(self):
        self._test(os.getcwd(), fs.pwd())


    # exists()
    def test_exists(self):
        self._test(True, fs.exists(__file__))
        self._test(True, fs.exists("/"))
        self._test(False, fs.exists("/not/a/real/path (I hope!)"))


    # isfile()
    def test_isfile(self):
        self._test(True, fs.isfile(__file__))
        self._test(False, fs.isfile("/"))
        self._test(False, fs.isfile("/not/a/real/path (I hope!)"))


    # isdir()
    def test_isdir(self):
        self._test(False, fs.isdir(__file__))
        self._test(True, fs.isdir("/"))
        self._test(False, fs.isdir("/not/a/real/path (I hope!)"))

    # read()
    def test_read_empty(self):
        self._test([],
                   fs.read("tests/data/empty_file"))

    def test_read_hello_world(self):
        self._test(['Hello, world!'],
                   fs.read("tests/data/hello_world"))

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
                   fs.read("tests/data/data1"))

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
                   fs.read("tests/data/data1", comment_char="#"))

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
                   fs.read("tests/data/data1", rstrip=False))

    # ls()
    def test_ls(self):
        pass

    # mkdir()
    def test_mkdir(self):
        fs.rm("/tmp/labm8.dir")
        self._test(False, fs.isdir("/tmp/labm8.dir"))
        fs.mkdir("/tmp/labm8.dir")
        self._test(True, fs.isdir("/tmp/labm8.dir"))

    def test_mkdir_parents(self):
        self._test(False, fs.isdir("/tmp/labm8.dir/foo/bar"))
        fs.mkdir("/tmp/labm8.dir/foo/bar")
        self._test(True, fs.isdir("/tmp/labm8.dir/foo/bar"))

    def test_mkdir_exists(self):
        fs.mkdir("/tmp/labm8.dir/")
        self._test(True, fs.isdir("/tmp/labm8.dir/"))
        fs.mkdir("/tmp/labm8.dir/")
        fs.mkdir("/tmp/labm8.dir/")
        self._test(True, fs.isdir("/tmp/labm8.dir/"))

    # mkopen()
    def test_mkopen(self):
        fs.rm("/tmp/labm8.dir")
        self._test(False, fs.isdir("/tmp/labm8.dir/"))
        f = fs.mkopen("/tmp/labm8.dir/foo", "w")
        self._test(True, fs.isdir("/tmp/labm8.dir/"))
        f.close()

    # rm()
    def test_rm(self):
        system.echo("Hello, world!", "/tmp/labm8.tmp")
        self._test(True, fs.isfile("/tmp/labm8.tmp"))
        fs.rm("/tmp/labm8.tmp")
        self._test(False, fs.isfile("/tmp/labm8.tmp"))
        fs.rm("/tmp/labm8.tmp")
        fs.rm("/tmp/labm8.tmp")
        fs.rm("/tmp/labm8.dir")
        fs.mkdir("/tmp/labm8.dir/foo/bar")
        system.echo("Hello, world!", "/tmp/labm8.dir/foo/bar/baz")
        self._test(True, fs.isfile("/tmp/labm8.dir/foo/bar/baz"))
        fs.rm("/tmp/labm8.dir")
        self._test(False, fs.isfile("/tmp/labm8.dir/foo/bar/baz"))
        self._test(False, fs.isfile("/tmp/labm8.dir/"))

    # cp()
    def test_cp(self):
        system.echo("Hello, world!", "/tmp/labm8.tmp")
        self._test(["Hello, world!"], fs.read("/tmp/labm8.tmp"))
        # Cleanup any existing file.
        fs.rm("/tmp/labm8.tmp.copy")
        self._test(False, fs.exists("/tmp/labm8.tmp.copy"))
        fs.cp("/tmp/labm8.tmp", "/tmp/labm8.tmp.copy")
        self._test(fs.read("/tmp/labm8.tmp"), fs.read("/tmp/labm8.tmp.copy"))

    def test_cp_no_file(self):
        self.assertRaises(IOError, fs.cp,
                          "/not a real src", "/not/a/real dest")

    def test_cp_dir(self):
        fs.rm("/tmp/labm8")
        fs.rm("/tmp/labm8.copy")
        fs.mkdir("/tmp/labm8/foo/bar")
        self._test(False, fs.exists("/tmp/labm8.copy"))
        fs.cp("/tmp/labm8/", "/tmp/labm8.copy")
        self._test(True, fs.isdir("/tmp/labm8.copy"))
        self._test(True, fs.isdir("/tmp/labm8.copy/foo"))
        self._test(True, fs.isdir("/tmp/labm8.copy/foo/bar"))

    def test_cp_overwrite(self):
        system.echo("Hello, world!", "/tmp/labm8.tmp")
        self._test(["Hello, world!"], fs.read("/tmp/labm8.tmp"))
        # Cleanup any existing file.
        fs.rm("/tmp/labm8.tmp.copy")
        self._test(False, fs.exists("/tmp/labm8.tmp.copy"))
        fs.cp("/tmp/labm8.tmp", "/tmp/labm8.tmp.copy")
        system.echo("Goodbye, world!", "/tmp/labm8.tmp")
        fs.cp("/tmp/labm8.tmp", "/tmp/labm8.tmp.copy")
        self._test(fs.read("/tmp/labm8.tmp"), fs.read("/tmp/labm8.tmp.copy"))


if __name__ == '__main__':
    main()
