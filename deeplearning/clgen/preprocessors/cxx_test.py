"""Unit tests for ///cxx_test."""
import sys

import pytest
from absl import app
from absl import flags

from deeplearning.clgen.preprocessors import cxx


FLAGS = flags.FLAGS


def test_ClangPreprocess_small_cxx_program():
  """Test pre-processing a small C++ program."""
  assert cxx.ClangPreprocess("""
#define FOO T
template<typename FOO>
FOO foobar(const T& a) {return a;}

int foo() { return foobar<int>(10); }
""") == """

template<typename T>
T foobar(const T& a) {return a;}

int foo() { return foobar<int>(10); }
"""


def test_StripComments_empty_input():
  """Test StripComments on an empty input."""
  assert cxx.StripComments('') == ''


def test_StripComments_only_commend():
  """Test StripComments on an empty input."""
  assert cxx.StripComments('// Just a comment') == ' '
  assert cxx.StripComments('/* Just a comment */') == ' '


def test_StripComments_small_program():
  """Test Strip Comments on a small program."""
  assert cxx.StripComments("""
/* comment header */

int main(int argc, char** argv) { // main function.
  return /* foo */ 0;
}
""") == """
 

int main(int argc, char** argv) {  
  return   0;
}
"""


def main(argv):
  """Main entry point."""
  del argv
  sys.exit(pytest.main([__file__, '-v']))


if __name__ == '__main__':
  app.run(main)
