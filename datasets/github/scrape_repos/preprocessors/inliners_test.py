"""Tests for //datasets/github.scrape_repos.preprocessors/inliners_test.py."""
import pathlib
import sys
import tempfile

import pytest
from absl import app
from absl import flags

from datasets.github.scrape_repos.preprocessors import inliners


FLAGS = flags.FLAGS


@pytest.fixture(scope='function')
def tempdir() -> pathlib.Path:
  """Test fixture for an empty temporary directory."""
  with tempfile.TemporaryDirectory() as d:
    yield pathlib.Path(d)


def MakeFile(directory: pathlib.Path, relpath: str, contents: str):
  """Write contents to a file."""
  abspath = (directory / relpath).absolute()
  abspath.parent.mkdir(parents=True, exist_ok=True)
  with open(abspath, 'w') as f:
    f.write(contents)


# CxxHeaders() tests.

def test_CxxHeaders_empty_file(tempdir: pathlib.Path):
  """Test that CxxHeaders() accepts an empty file."""
  (tempdir / 'a').touch()
  assert inliners.CxxHeaders(tempdir, 'a', '') == ''


def test_CxxHeaders_no_includes(tempdir: pathlib.Path):
  """Test that CxxHeaders() doesn't modify a file without includes."""
  src = """
int main(int argc, char** argv) {
  return 0;
}   
"""
  (tempdir / 'a').touch()
  assert inliners.CxxHeaders(tempdir, 'a', src) == src


def test_CxxHeaders_subdir_no_includes(tempdir: pathlib.Path):
  """CxxHeaders() doesn't modify a file in a subdir without includes."""
  src = """
int main(int argc, char** argv) {
  return 0;
}   
"""
  MakeFile(tempdir / 'foo', 'a', src)
  assert inliners.CxxHeaders(tempdir, 'foo/a', src) == src


def test_CxxHeaders_header_in_same_dir(tempdir: pathlib.Path):
  """CxxHeaders() inlines a file from the same directory."""
  src = """
#include "foo.h"

int main(int argc, char** argv) { return 0; }
"""
  MakeFile(tempdir, 'a', src)
  MakeFile(tempdir, 'foo.h', '#define FOO')
  assert inliners.CxxHeaders(tempdir, 'a', src) == """
// [InlineHeaders] Found candidate include for: 'foo.h' -> 'foo.h'.
#define FOO

int main(int argc, char** argv) { return 0; }
"""


def test_CxxHeaders_no_match(tempdir: pathlib.Path):
  """CxxHeaders() preserves an include with no match."""
  src = """
#include "foo.h"

int main(int argc, char** argv) { return 0; }
"""
  MakeFile(tempdir, 'a', src)
  assert inliners.CxxHeaders(tempdir, 'a', src) == """
// [InlineHeaders] Preserving unmatched include: 'foo.h'.
#include "foo.h"

int main(int argc, char** argv) { return 0; }
"""


def test_CxxHeaders_ignore_system_headers(tempdir: pathlib.Path):
  """CxxHeaders() ignores system headers."""
  src = """
#include <stdio.h>

int main(int argc, char** argv) { return 0; }
"""
  MakeFile(tempdir, 'a', src)
  # Note that the angle brackets have been re-written with quotes.
  assert inliners.CxxHeaders(tempdir, 'a', src) == """
// [InlineHeaders] Preserving blacklisted include: 'stdio.h'.
#include "stdio.h"

int main(int argc, char** argv) { return 0; }
"""


# CxxHeadersDiscardUnknown() tests.


def test_CxxHeadersDiscardUnknown_no_match(tempdir: pathlib.Path):
  """CxxHeadersDiscardUnknown() discards an include with no match."""
  src = """
#include "foo.h"

int main(int argc, char** argv) { return 0; }
"""
  MakeFile(tempdir, 'a', src)
  assert inliners.CxxHeadersDiscardUnknown(tempdir, 'a', src) == """
// [InlineHeaders] Discarding unmatched include: 'foo.h'.

int main(int argc, char** argv) { return 0; }
"""


# GetAllFilesRelativePaths() tests.

def test_GetAllFilesRelativePaths_empty_dir(tempdir: pathlib.Path):
  """Test that an empty directory returns an empty list."""
  assert inliners.GetAllFilesRelativePaths(tempdir) == []


def test_GetAllFilesRelativePaths_relpath(tempdir: pathlib.Path):
  """Test that relative paths are returned."""
  (tempdir / 'a').touch()
  assert inliners.GetAllFilesRelativePaths(tempdir) == ['a']


# GetLibCxxHeaders() tests.

def test_GetLibCxxHeaders():
  headers = inliners.GetLibCxxHeaders()
  assert 'stdio.h' in headers
  assert 'string' in headers


def main(argv):
  """Main entry point."""
  if len(argv) > 1:
    raise app.UsageError("Unknown arguments: '{}'.".format(' '.join(argv[1:])))
  sys.exit(pytest.main([__file__, '-vv']))


if __name__ == '__main__':
  flags.FLAGS(['argv[0]', '-v=1'])
  app.run(main)
