# This file is part of libcecl.
#
# Copyright (c) 2016, 2017, 2018, 2019 Chris Cummins.
#
# libcecl is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# libcecl is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with libcecl.  If not, see <https://www.gnu.org/licenses/>.
"""Unit tests for //gpu/libcecl:libcecl_rewriter."""
from gpu.libcecl import libcecl_rewriter
from labm8.py import test


def test_RewriteOpenClSource_adds_header():
  """Short summary of test."""
  assert libcecl_rewriter.RewriteOpenClSource("""\
#include <CL/cl.h>

int main() {
}
""") == """\
#include <libcecl.h>
#include <CL/cl.h>

int main() {
}
"""


if __name__ == '__main__':
  test.Main()
