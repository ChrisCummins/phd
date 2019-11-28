"""Unit tests for //experimental/deeplearning/deepsmith/java_fuzz:preprocess_java_corpus."""
from datasets.github.scrape_repos import contentfiles
from experimental.deeplearning.deepsmith.java_fuzz import preprocess_java_corpus
from labm8.py import test

FLAGS = test.FLAGS


def test_PreprocessContentFiles():
  """Test preprocessing a basic input."""
  pp_cfs = preprocess_java_corpus.PreprocessContentFiles(
    [
      contentfiles.ContentFile(
        text="""
private static int Foobar(int foo) {
          int bar = 10    + 1; foo += bar;
                foo *= 2;
          return foo + 10;
        }
"""
      )
    ]
  )
  assert len(pp_cfs) == 1
  assert (
    pp_cfs[0].text
    == """\
private static int fn_A(int a){
  int b=10 + 1;
  a+=b;
  a*=2;
  return a + 10;
}
"""
  )
  assert pp_cfs[0].preprocessing_succeeded


def test_PreprocessContentFiles_method_depends_on_java_util():
  """Test that a method which uses java.util.ArrayList works."""
  pp_cfs = preprocess_java_corpus.PreprocessContentFiles(
    [
      contentfiles.ContentFile(
        text="""
private static int Foobar(int a, ArrayList<Integer> _) {
  int b=10 + 1;
  a+=b;
  a*=2;
  return a + 10;
}
"""
      )
    ]
  )
  assert len(pp_cfs) == 1
  assert (
    pp_cfs[0].text
    == """\
private static int fn_A(int a,ArrayList<Integer> b){
  int c=10 + 1;
  a+=c;
  a*=2;
  return a + 10;
}
"""
  )
  assert pp_cfs[0].preprocessing_succeeded


if __name__ == "__main__":
  test.Main()
