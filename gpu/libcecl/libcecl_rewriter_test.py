"""Unit tests for //gpu/libcecl:libcecl_rewriter."""
from gpu.libcecl import libcecl_rewriter
from labm8 import test


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
