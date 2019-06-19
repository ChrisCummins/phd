"""Unit tests for //experimental/deeplearning/deepsmith/java_fuzz:preprocess_java_corpus."""

from datasets.github.scrape_repos import contentfiles

from experimental.deeplearning.deepsmith.java_fuzz import preprocess_java_corpus
from labm8 import test

FLAGS = test.FLAGS


def test_PreprocessList():
  """Test preprocessing a basic input."""
  pp_cfs = preprocess_java_corpus.PreprocessList([
      contentfiles.ContentFile(text="""
private static int Foobar(int foo) {
          int bar = 10    + 1; foo += bar;
                foo *= 2;
          return foo + 10;
        }
""")
  ])

  assert len(pp_cfs) == 1
  assert pp_cfs[0].text == """\
private static int fn_A(int a){
  int b=10 + 1;
  a+=b;
  a*=2;
  return a + 10;
}
"""


if __name__ == '__main__':
  test.Main()
