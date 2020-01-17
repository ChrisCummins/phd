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


def MakeFiles(relpaths):
  """Create the given list of paths relative to the current directory."""
  for path in relpaths:
    path = pathlib.Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    path.touch()


def test_GeneratePaths_non_existent_path(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  paths = list(path_generator.GeneratePaths([str(tempdir / "not_a_path")]))
  assert paths == []


def test_GeneratePaths_single_abspath(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  MakeFiles(
    [tempdir / "hello.txt",]
  )

  paths = list(path_generator.GeneratePaths([str(tempdir / "hello.txt")]))

  assert paths == [tempdir / "hello.txt"]


def test_GeneratePaths_single_relpath(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  os.chdir(tempdir)

  MakeFiles(
    [tempdir / "hello.txt",]
  )

  paths = list(path_generator.GeneratePaths(["hello.txt"]))

  assert paths == [
    tempdir / "hello.txt",
  ]


def test_GeneratePaths_empty_directory(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  paths = list(path_generator.GeneratePaths([str(tempdir)]))
  assert paths == []


def test_GeneratePaths_directory_with_file(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  MakeFiles(
    [tempdir / "a",]
  )

  paths = list(path_generator.GeneratePaths([str(tempdir)]))

  assert paths == [
    tempdir / "a",
  ]


def test_GeneratePaths_file_in_ignore_list(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  MakeFiles(
    [tempdir / ".formatignore", tempdir / "a",]
  )
  fs.Write(tempdir / ".formatignore", "a".encode("utf-8"))

  paths = list(path_generator.GeneratePaths([str(tempdir)]))

  assert paths == [tempdir / ".formatignore"]


def test_GeneratePaths_ignore_list_glob(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  MakeFiles(
    [tempdir / ".formatignore", tempdir / "a",]
  )
  fs.Write(tempdir / ".formatignore", "*".encode("utf-8"))
  paths = list(path_generator.GeneratePaths([str(tempdir)]))

  assert paths == [tempdir / ".formatignore"]


def test_GeneratePaths_ignore_list_glob_dot_files(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  MakeFiles(
    [tempdir / ".formatignore", tempdir / "a",]
  )
  fs.Write(tempdir / ".formatignore", ".*".encode("utf-8"))

  paths = list(path_generator.GeneratePaths([str(tempdir)]))

  assert paths == [
    tempdir / "a",
  ]


def test_GeneratePaths_ignore_list_glob_unignored(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  os.chdir(tempdir)
  MakeFiles(
    [".formatignore", "a", "b", "c",]
  )
  fs.Write(".formatignore", "*\n!a".encode("utf-8"))

  paths = list(path_generator.GeneratePaths(["."]))

  assert paths == [
    tempdir / ".formatignore",
    tempdir / "a",
  ]


def test_GeneratePaths_ignore_list_parent_directory(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  """Test ignoring the contents of a directory."""
  os.chdir(tempdir)
  MakeFiles(
    [".formatignore", "src/a", "src/b",]
  )
  fs.Write(".formatignore", "src".encode("utf-8"))

  paths = list(path_generator.GeneratePaths(["."]))

  assert paths == [tempdir / ".formatignore"]


def test_GeneratePaths_ignore_list_glob_parent_directory(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  """Test ignoring the contents of a directory.

  File layout:

    /.formatignore -> "*"
    /src
    /src/a
  """
  os.chdir(tempdir)
  MakeFiles(
    [".formatignore", "src/a", "src/b",]
  )
  fs.Write(".formatignore", "*".encode("utf-8"))

  paths = list(path_generator.GeneratePaths(["."]))

  assert paths == [tempdir / ".formatignore"]


def test_GeneratePaths_ignore_list_recurisve_glob(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  """Test ignoring files in a recursive glob."""
  os.chdir(tempdir)
  MakeFiles(
    [".formatignore", "src/a", "src/b", "src/c/a/a", "src/c/a/b",]
  )
  fs.Write(".formatignore", "**/a".encode("utf-8"))

  paths = list(path_generator.GeneratePaths(["."]))
  print(paths)

  assert paths == [
    tempdir / ".formatignore",
    tempdir / "src/b",
  ]


def test_GeneratePaths_ignore_git_submodule(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  """Test that git submodules are not visited."""
  os.chdir(tempdir)
  MakeFiles(
    [
      ".git/config",  # Fake repo root, should be ignored
      "README",
      "src/a",
      "src/b",
      "src/c/d",
      "src/submod/.git",  # Fake submodule, should be ignored
      "src/submod/a",  # should be ignored
      "src/submod/b",  # should be ignored
      "src/submod/c/c",  # should be ignored
    ]
  )

  paths = set(path_generator.GeneratePaths(["."]))
  assert paths == {
    tempdir / "README",
    tempdir / "src/a",
    tempdir / "src/b",
    tempdir / "src/c/d",
  }


def test_GeneratePaths_explicitly_requested_submodule(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  """Test that a git submodule is visited if it is explicitly asked for."""
  os.chdir(tempdir)
  MakeFiles(
    [
      ".git/config",  # Fake repo root, should be ignored
      "README",
      "src/a",
      "src/b",
      "src/c/d",
      "src/submod/.git",  # Fake submodule, should be ignored
      "src/submod/a",  # should be ignored
      "src/submod/b",  # should be ignored
      "src/submod/c/c",  # should be ignored
    ]
  )

  paths = set(path_generator.GeneratePaths(["src/submod"]))
  assert paths == {
    tempdir / "src/submod/a",
    tempdir / "src/submod/b",
    tempdir / "src/submod/c/c",
  }


def test_GeneratePaths_ignored_in_glob_expansion(
  path_generator: path_generator_lib.PathGenerator, tempdir: pathlib.Path
):
  """Test that a git submodule is not visited if it would only be visited as
  the result of a glob expansion.
  """
  os.chdir(tempdir)
  MakeFiles(
    [
      ".git/config",  # Fake repo root, should be ignored
      "README",
      "src/a",
      "src/b",
      "src/c/d",
      "src/submod/.git",  # Fake submodule, should be ignored
      "src/submod/a",  # should be ignored
      "src/submod/b",  # should be ignored
      "src/submod/c/c",  # should be ignored
    ]
  )

  paths = list(path_generator.GeneratePaths(["src/*"]))

  assert paths == [
    tempdir / "src/a",
    tempdir / "src/b",
    tempdir / "src/c/d",
  ]


if __name__ == "__main__":
  test.Main()
