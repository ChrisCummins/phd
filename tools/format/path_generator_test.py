# Copyright 2020 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for //tools/format:path_generator."""
import os
import pathlib

from labm8.py import fs
from labm8.py import test
from tools.format import path_generator as path_generator_lib

FLAGS = test.FLAGS


@test.Fixture(scope="function")
def path_generator():
  return path_generator_lib.PathGenerator(".formatignore")


def test_GeneratePaths_non_existent_path(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  paths = list(path_generator.GeneratePaths([str(tempdir / "not_a_path")]))
  assert paths == []


def test_GeneratePaths_single_abspath(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  path = tempdir / "hello.txt"
  path.touch()
  paths = list(path_generator.GeneratePaths([str(path)]))
  assert paths == [path]


def test_GeneratePaths_single_relpath(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  os.chdir(tempdir)
  path = tempdir / "hello.txt"
  path.touch()
  paths = list(path_generator.GeneratePaths([path.name]))
  assert paths == [path]


def test_GeneratePaths_empty_directory(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  paths = list(path_generator.GeneratePaths([str(tempdir)]))
  assert paths == []


def test_GeneratePaths_directory_with_file(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  path = tempdir / "a"
  path.touch()
  paths = list(path_generator.GeneratePaths([str(tempdir)]))
  assert paths == [path]


def test_GeneratePaths_file_in_ignore_list(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  path = tempdir / "a"
  ignore_file = tempdir / ".formatignore"
  path.touch()
  fs.Write(ignore_file, "a".encode("utf-8"))
  paths = list(path_generator.GeneratePaths([str(tempdir)]))
  assert paths == [ignore_file]


def test_GeneratePaths_ignore_list_glob(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  path = tempdir / "a"
  ignore_file = tempdir / ".formatignore"
  path.touch()
  fs.Write(ignore_file, "*".encode("utf-8"))
  paths = list(path_generator.GeneratePaths([str(tempdir)]))
  assert paths == [ignore_file]


def test_GeneratePaths_ignore_list_glob_hidden_files(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  path = tempdir / "a"
  ignore_file = tempdir / ".formatignore"
  path.touch()
  fs.Write(ignore_file, ".*".encode("utf-8"))
  paths = list(path_generator.GeneratePaths([str(tempdir)]))
  assert paths == [path]


def test_GeneratePaths_ignore_list_glob_unignored(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  path = tempdir / "a"
  ignore_file = tempdir / ".formatignore"
  path.touch()
  fs.Write(ignore_file, "*\n!a".encode("utf-8"))
  paths = set(path_generator.GeneratePaths([str(tempdir)]))
  assert paths == {path, ignore_file}


def test_GeneratePaths_ignore_list_parent_directory(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  """Test ignoring the contents of a directory.

  File layout:

    /.formatignore -> "src"
    /src
    /src/a

  Expected outcome is that "a" should be ignored because it belongs in src.
  """
  parent = tempdir / "src"
  parent.mkdir()
  path = parent / "a"
  ignore_file = tempdir / ".formatignore"
  path.touch()
  fs.Write(ignore_file, "src".encode("utf-8"))
  paths = list(path_generator.GeneratePaths([str(tempdir)]))
  assert paths == [ignore_file]


def test_GeneratePaths_ignore_list_glob_parent_directory(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  """Test ignoring the contents of a directory.

  File layout:

    /.formatignore -> "*"
    /src
    /src/a
  """
  parent = tempdir / "src"
  parent.mkdir()
  path = parent / "a"
  ignore_file = tempdir / ".formatignore"
  path.touch()
  fs.Write(ignore_file, "*".encode("utf-8"))
  paths = list(path_generator.GeneratePaths([str(tempdir)]))
  assert paths == [ignore_file]


def test_GeneratePaths_ignore_list_recurisve_glob(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  """Test ignoring files in a subdirectory.

  File layout:

    /.formatignore -> "**/a"
    /src
    /src/a
    /src/b
  """
  parent = tempdir / "src"
  parent.mkdir()
  a = parent / "a"
  b = parent / "b"
  ignore_file = tempdir / ".formatignore"
  a.touch()
  b.touch()
  fs.Write(ignore_file, "**/a".encode("utf-8"))
  paths = set(path_generator.GeneratePaths([str(tempdir)]))
  assert paths == {ignore_file, b}


if __name__ == "__main__":
  test.Main()
