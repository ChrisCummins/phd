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
        self._test(os.path.expanduser("~"), fs.path("~"))
        self._test(os.path.join(os.path.expanduser("~"), "foo"),
                   fs.path("~", "foo"))

    def test_must_exist(self):
        system.echo("Hello, world!", "/tmp/labm8.must_exist.txt")
        self.assertEqual(fs.must_exist("/tmp/labm8.must_exist.txt"),
                         "/tmp/labm8.must_exist.txt")
        self.assertEqual(fs.must_exist("/tmp", "labm8.must_exist.txt"),
                         "/tmp/labm8.must_exist.txt")
        with self.assertRaises(fs.File404):
            fs.must_exist("/not/a/real/path")
        fs.rm("/tmp/labm8.must_exist.txt")


    # abspath()
    def test_abspath(self):
        self._test(os.path.abspath(".") + "/foo/bar",
                   fs.abspath("foo", "bar"))
        self._test(os.path.abspath(".") + "/foo/bar/car",
                   fs.abspath("foo/bar", "car"))

    def test_abspath_homedir(self):
        self._test(os.path.expanduser("~"), fs.abspath("~"))
        self._test(os.path.join(os.path.expanduser("~"), "foo"),
                   fs.abspath("~", "foo"))

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
        new = fs.abspath("..")

        self._test(new, fs.cd(".."))
        self._test(new, os.getcwd())

        self._test(cwd, fs.cdpop())
        self._test(cwd, os.getcwd())

        self._test(cwd, fs.cdpop())
        self._test(cwd, os.getcwd())

        self._test(cwd, fs.cdpop())
        self._test(cwd, os.getcwd())


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

    # isexe()
    def test_isexe(self):
        self._test(True, fs.isexe("/bin/ls"))
        self._test(False, fs.isexe("/home"))
        self._test(False, fs.isexe("/not/a/real/path (I hope!)"))

    # isdir()
    def test_isdir(self):
        self._test(False, fs.isdir(__file__))
        self._test(True, fs.isdir("/"))
        self._test(False, fs.isdir("/not/a/real/path (I hope!)"))

    # read()
    def test_read(self):
        self._test(['Hello, world!'],
                   fs.read("tests/data/hello_world"))
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

    def test_read_no_rstrip(self):
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

    def test_read_ignore_comments(self):
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

    def test_read_ignore_comments_no_rstrip(self):
        self._test([
            'This\n',
            'is a test file\n',
            'With\n',
            'trailing  ',
            '\n',
            '\n',
            '\n',
            'whitespace\n',
            '0.344\n'
        ],
                   fs.read("tests/data/data1",
                           rstrip=False, comment_char="#"))

    def test_read_empty_file(self):
        self._test([],
                   fs.read("tests/data/empty_file"))

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

    def test_rm_glob(self):
        fs.mkdir("/tmp/labm8.glob")
        system.echo("Hello, world!", "/tmp/labm8.glob/1")
        system.echo("Hello, world!", "/tmp/labm8.glob/2")
        system.echo("Hello, world!", "/tmp/labm8.glob/abc")

        fs.rm("/tmp/labm8.glob/a*", glob=False)
        self._test(True, fs.isfile("/tmp/labm8.glob/1"))
        self._test(True, fs.isfile("/tmp/labm8.glob/2"))
        self._test(True, fs.isfile("/tmp/labm8.glob/abc"))

        fs.rm("/tmp/labm8.glob/a*")
        self._test(True, fs.isfile("/tmp/labm8.glob/1"))
        self._test(True, fs.isfile("/tmp/labm8.glob/2"))
        self._test(False, fs.isfile("/tmp/labm8.glob/abc"))

        fs.rm("/tmp/labm8.glob/*")
        self._test(False, fs.isfile("/tmp/labm8.glob/1"))
        self._test(False, fs.isfile("/tmp/labm8.glob/2"))
        self._test(False, fs.isfile("/tmp/labm8.glob/abc"))

    # rmtrash()
    def test_rmtrash(self):
        system.echo("Hello, world!", "/tmp/labm8.tmp")
        self.assertTrue(fs.isfile("/tmp/labm8.tmp"))
        fs.rmtrash("/tmp/labm8.tmp")
        self.assertFalse(fs.isfile("/tmp/labm8.tmp"))
        fs.rmtrash("/tmp/labm8.tmp")
        fs.rm("/tmp/labm8.tmp")
        fs.rm("/tmp/labm8.dir")
        fs.mkdir("/tmp/labm8.dir/foo/bar")
        system.echo("Hello, world!", "/tmp/labm8.dir/foo/bar/baz")
        self.assertTrue(fs.isfile("/tmp/labm8.dir/foo/bar/baz"))
        fs.rmtrash("/tmp/labm8.dir")
        self.assertFalse(fs.isfile("/tmp/labm8.dir/foo/bar/baz"))
        self.assertFalse(fs.isfile("/tmp/labm8.dir/"))

    def test_rmtrash_bad_path(self):
        fs.rmtrash("/not/a/real/path")

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

    def test_cp_over_dir(self):
        fs.mkdir("/tmp/labm8.tmp.src")
        system.echo("Hello, world!", "/tmp/labm8.tmp.src/foo")
        fs.rm("/tmp/labm8.tmp.copy")
        fs.mkdir("/tmp/labm8.tmp.copy")
        self._test(True, fs.isdir("/tmp/labm8.tmp.src"))
        self._test(True, fs.isfile("/tmp/labm8.tmp.src/foo"))
        self._test(True, fs.isdir("/tmp/labm8.tmp.copy"))
        self._test(False, fs.isfile("/tmp/labm8.tmp.copy/foo"))
        fs.cp("/tmp/labm8.tmp.src", "/tmp/labm8.tmp.copy/")
        self._test(True, fs.isdir("/tmp/labm8.tmp.src"))
        self._test(True, fs.isfile("/tmp/labm8.tmp.src/foo"))
        self._test(True, fs.isdir("/tmp/labm8.tmp.copy"))
        self._test(True, fs.isfile("/tmp/labm8.tmp.copy/foo"))
        self._test(fs.read("/tmp/labm8.tmp.src/foo"),
                   fs.read("/tmp/labm8.tmp.copy/foo"))

    # mv()
    def test_mv(self):
        system.echo("Hello, world!", "/tmp/labm8.tmp")
        self._test(["Hello, world!"], fs.read("/tmp/labm8.tmp"))
        # Cleanup any existing file.
        fs.rm("/tmp/labm8.tmp.copy")
        self._test(False, fs.exists("/tmp/labm8.tmp.copy"))
        fs.mv("/tmp/labm8.tmp", "/tmp/labm8.tmp.copy")
        self.assertEqual(["Hello, world!"], fs.read("/tmp/labm8.tmp.copy"))
        self._test(False, fs.exists("/tmp/labm8.tmp"))

    def test_mv_no_src(self):
        with self.assertRaises(fs.File404):
            fs.mv("/bad/path", "foo")

    def test_mv_no_dst(self):
        system.echo("Hello, world!", "/tmp/labm8.tmp")
        with self.assertRaises(IOError):
            fs.mv("/tmp/labm8.tmp", "/not/a/real/path")
        fs.rm("/tmp/labm8.tmp")

    # ls()
    def test_ls(self):
        self._test(["a", "b", "c", "d"],
                   fs.ls("tests/data/testdir"))

    def test_ls_recursive(self):
        self._test(["a", "b", "c", "c/e", "c/f", "c/f/f",
                    "c/f/f/i", "c/f/h", "c/g", "d"],
                   fs.ls("tests/data/testdir", recursive=True))

    def test_ls_abspaths(self):
        fs.cp("tests/data/testdir", "/tmp/testdir")
        self._test(["/tmp/testdir/a",
                    "/tmp/testdir/b",
                    "/tmp/testdir/c",
                    "/tmp/testdir/d"],
                   fs.ls("/tmp/testdir", abspaths=True))
        self._test(["/tmp/testdir/a",
                    "/tmp/testdir/b",
                    "/tmp/testdir/c",
                    "/tmp/testdir/c/e",
                    "/tmp/testdir/c/f",
                    "/tmp/testdir/c/f/f",
                    "/tmp/testdir/c/f/f/i",
                    "/tmp/testdir/c/f/h",
                    "/tmp/testdir/c/g",
                    "/tmp/testdir/d"],
                   fs.ls("/tmp/testdir", recursive=True, abspaths=True))
        fs.rm("/tmp/testdir")

    def test_ls_empty_dir(self):
        fs.mkdir("/tmp/labm8.empty")
        self._test([], fs.ls("/tmp/labm8.empty"))
        fs.rm("/tmp/labm8.empty")

    def test_ls_bad_path(self):
        with self.assertRaises(OSError):
            fs.ls("/not/a/real/path/bro")

    def test_ls_single_file(self):
        self._test(["a"], fs.ls("tests/data/testdir/a"))

    # lsdirs()
    def test_lsdirs(self):
        self._test(["c"], fs.lsdirs("tests/data/testdir"))

    def test_lsdirs_recursive(self):
        self._test(["c", "c/f", "c/f/f"],
                   fs.lsdirs("tests/data/testdir", recursive=True))

    def test_lsdirs_bad_path(self):
        with self.assertRaises(OSError):
            fs.lsdirs("/not/a/real/path/bro")

    def test_lsdirs_single_file(self):
        self._test([], fs.lsdirs("tests/data/testdir/a"))

    # lsdirs()
    def test_lsfiles(self):
        self._test(["a", "b", "d"], fs.lsfiles("tests/data/testdir"))

    def test_lsfiles_recursive(self):
        self._test(["a", "b", "c/e", "c/f/f/i", "c/f/h", "c/g", "d"],
                   fs.lsfiles("tests/data/testdir", recursive=True))

    def test_lsfiles_bad_path(self):
        with self.assertRaises(OSError):
            fs.lsfiles("/not/a/real/path/bro")

    def test_lsfiles_single_file(self):
        self._test(["a"], fs.lsfiles("tests/data/testdir/a"))
