"""Test for clgen_preprocess image."""
import pathlib

import pytest

from labm8.py import dockerutil
from labm8.py import fs
from labm8.py import test

FLAGS = test.FLAGS

MODULE_UNDER_TEST = None


@test.Fixture(scope="function")
def contentfiles(tempdir: pathlib.Path) -> pathlib.Path:
  fs.Write(tempdir / "a.cc", "int main() /* comment */ {}".encode("utf-8"))
  fs.Write(
    tempdir / "b.java",
    """
public class HelloWorld {
  public static void main(String[] args) {
    System.out.println("Hello, world!");
  }
}
""".encode(
      "utf-8"
    ),
  )
  yield tempdir


@test.Fixture(scope="function")
def preprocess() -> dockerutil.BazelPy3Image:
  return dockerutil.BazelPy3Image("deeplearning/clgen/docker/clgen_preprocess")


def test_preprocess_image_smoke_test(preprocess: dockerutil.BazelPy3Image):
  """Check that image doesn't blow up."""
  with preprocess.RunContext() as ctx:
    ctx.CheckCall(["--version"], timeout=30)


def test_cxx_preprocess(
  contentfiles: pathlib.Path,
  tempdir2: pathlib.Path,
  preprocess: dockerutil.BazelPy3Image,
):
  """Test pre-processing C++ contentfiles"""
  # This is functionally the same as the test in
  # //deeplearning/clgen:preprocess_test.
  with preprocess.RunContext() as ctx:
    ctx.CheckCall(
      [],
      {
        "contentfiles": "/contentfiles",
        "outdir": "/preprocessed",
        "preprocessors": (
          "deeplearning.clgen.preprocessors.cxx:Compile,"
          "deeplearning.clgen.preprocessors.cxx:NormalizeIdentifiers,"
          "deeplearning.clgen.preprocessors.cxx:StripComments,"
          "deeplearning.clgen.preprocessors.cxx:ClangFormat"
        ),
      },
      volumes={contentfiles: "/contentfiles", tempdir2: "/preprocessed"},
    )

  assert (tempdir2 / "a.cc").is_file()
  assert not (tempdir2 / "b.java").is_file()

  with open(tempdir2 / "a.cc") as f:
    assert (
      f.read()
      == """int A() {
}"""
    )


def test_java_preprocess(
  contentfiles: pathlib.Path,
  tempdir2: pathlib.Path,
  preprocess: dockerutil.BazelPy3Image,
):
  """Test pre-processing Java contentfiles"""
  with preprocess.RunContext() as ctx:
    ctx.CheckCall(
      [],
      {
        "contentfiles": "/contentfiles",
        "outdir": "/preprocessed",
        "preprocessors": "deeplearning.clgen.preprocessors.java:Compile",
      },
      volumes={contentfiles: "/contentfiles", tempdir2: "/preprocessed"},
    )

  assert not (tempdir2 / "a.cc").is_file()
  assert (tempdir2 / "b.java").is_file()

  with open(tempdir2 / "b.java") as f:
    assert (
      f.read()
      == """
public class HelloWorld {
  public static void main(String[] args) {
    System.out.println("Hello, world!");
  }
}
"""
    )


if __name__ == "__main__":
  test.Main()
