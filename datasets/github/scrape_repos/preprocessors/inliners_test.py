# Copyright 2018, 2019 Chris Cummins <chrisc.101@gmail.com>.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for //datasets/github.scrape_repos.preprocessors/inliners_test.py."""
import pathlib

from datasets.github.scrape_repos.preprocessors import inliners
from labm8.py import app
from labm8.py import test

FLAGS = app.FLAGS


def MakeFile(directory: pathlib.Path, relpath: str, contents: str):
  """Write contents to a file."""
  abspath = (directory / relpath).absolute()
  abspath.parent.mkdir(parents=True, exist_ok=True)
  with open(abspath, "w") as f:
    f.write(contents)


# CxxHeaders() tests.


def test_CxxHeaders_empty_file(tempdir: pathlib.Path):
  """Test that CxxHeaders() accepts an empty file."""
  (tempdir / "a").touch()
  assert inliners.CxxHeaders(tempdir, "a", "", ["a"]) == [""]


def test_CxxHeaders_no_includes(tempdir: pathlib.Path):
  """Test that CxxHeaders() doesn't modify a file without includes."""
  src = """
int main(int argc, char** argv) {
  return 0;
}
"""
  (tempdir / "a").touch()
  assert inliners.CxxHeaders(tempdir, "a", src, ["a"]) == [src]


def test_CxxHeaders_subdir_no_includes(tempdir: pathlib.Path):
  """CxxHeaders() doesn't modify a file in a subdir without includes."""
  src = """
int main(int argc, char** argv) {
  return 0;
}
"""
  MakeFile(tempdir / "foo", "a", src)
  assert inliners.CxxHeaders(tempdir, "foo/a", src, ["foo/a"]) == [src]


def test_CxxHeaders_header_in_same_dir(tempdir: pathlib.Path):
  """CxxHeaders() inlines a file from the same directory."""
  src = """
#include "foo.h"

int main(int argc, char** argv) { return 0; }
"""
  MakeFile(tempdir, "a", src)
  MakeFile(tempdir, "foo.h", "#define FOO")
  assert inliners.CxxHeaders(tempdir, "a", src, ["a", "foo.h"]) == [
    """
// [InlineHeaders] Found candidate include for: 'foo.h' -> 'foo.h' (100% confidence).
#define FOO

int main(int argc, char** argv) { return 0; }
"""
  ]


def test_CxxHeaders_no_match(tempdir: pathlib.Path):
  """CxxHeaders() preserves an include with no match."""
  src = """
#include "foo.h"

int main(int argc, char** argv) { return 0; }
"""
  MakeFile(tempdir, "a", src)
  assert inliners.CxxHeaders(tempdir, "a", src, ["a"]) == [
    """
// [InlineHeaders] Preserving unmatched include: 'foo.h'.
#include "foo.h"

int main(int argc, char** argv) { return 0; }
"""
  ]


def test_CxxHeaders_ignore_libcxx_headers(tempdir: pathlib.Path):
  """CxxHeaders() ignores libcxx headers."""
  src = """
#include <cassert>

int main(int argc, char** argv) { return 0; }
"""
  MakeFile(tempdir, "a", src)
  # Note that the angle brackets have been re-written with quotes.
  assert inliners.CxxHeaders(tempdir, "a", src, ["a"]) == [
    """
// [InlineHeaders] Preserving blacklisted include: 'cassert'.
#include "cassert"

int main(int argc, char** argv) { return 0; }
"""
  ]


def test_CxxHeaders_fuzzy_match(tempdir: pathlib.Path):
  """CxxHeaders() fuzzy matches the closest candidate include."""
  src = """
#include <proj/foo/foo.h>

int main(int argc, char** argv) { return 0; }
"""
  MakeFile(tempdir / "src" / "proj", "src.c", src)
  MakeFile(tempdir / "src" / "proj" / "foo", "foo.h", "#define FOO")
  MakeFile(tempdir / "bar" / "foo" / "proj" / "foo", "foo.h", "#define NOT_FOO")
  assert (
    [
      """
// [InlineHeaders] Found candidate include for: 'proj/foo/foo.h' -> \
'src/proj/foo/foo.h' (95% confidence).
#define FOO

int main(int argc, char** argv) { return 0; }
"""
    ]
    == inliners.CxxHeaders(
      tempdir,
      "src/proj/src.c",
      src,
      ["src/proj/src.c", "src/proj/foo/foo.h", "bar/foo/proj/foo/foo.h",],
    )
  )


# CxxHeadersDiscardUnknown() tests.


def test_CxxHeadersDiscardUnknown_no_match(tempdir: pathlib.Path):
  """CxxHeadersDiscardUnknown() discards an include with no match."""
  src = """
#include "foo.h"

int main(int argc, char** argv) { return 0; }
"""
  MakeFile(tempdir, "a", src)
  assert inliners.CxxHeadersDiscardUnknown(tempdir, "a", src, ["a"]) == [
    """
// [InlineHeaders] Discarding unmatched include: 'foo.h'.

int main(int argc, char** argv) { return 0; }
"""
  ]


# GetLibCxxHeaders() tests.


def test_GetLibCxxHeaders():
  headers = inliners.GetLibCxxHeaders()
  assert "stdio.h" in headers
  assert "string" in headers


if __name__ == "__main__":
  test.Main()
