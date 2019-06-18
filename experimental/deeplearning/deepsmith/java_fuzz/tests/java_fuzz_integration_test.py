"""Integration test for Java fuzz pipeline."""

import pathlib

from labm8 import dockerutil
from labm8 import test

FLAGS = test.FLAGS

MODULE_UNDER_TEST = None  # Disable coverage.


def test_end_to_end_pipeline(tempdir: pathlib.Path):
  scrape_java_files_image = dockerutil.BazelPy3Image(
      'experimental/deeplearning/deepsmith/java_fuzz/scrape_java_files_image')
  mask_contentfiles_image = dockerutil.BazelPy3Image(
      'experimental/deeplearning/deepsmith/java_fuzz/mask_contentfiles_image')
  export_java_corpus_image = dockerutil.BazelPy3Image(
      'experimental/deeplearning/deepsmith/java_fuzz/export_java_corpus_image')

  # Step 1: Scrape a single repo from GitHub.
  with scrape_java_files_image.RunContext() as ctx:
    ctx.CheckCall([], {
        "n": 1,
        "db": 'sqlite:////workdir/java.db',
    },
                  volumes={
                      tempdir: '/workdir',
                      '/var/phd': '/var/phd',
                  })

  # Check that contentfiles database is created.
  assert (tempdir / "java.db").is_file()

  # Step 2: Mask a subset of the contentfiles database.
  with mask_contentfiles_image.RunContext() as ctx:
    ctx.CheckCall([], {
        "db": 'sqlite:////workdir/java.db',
        "max_repo_count": 1,
    },
                  volumes={tempdir: '/workdir'})

  # Check that contentfiles database is still there.
  assert (tempdir / "java.db").is_file()

  # Step 3: Export Java corpus.
  with export_java_corpus_image.RunContext() as ctx:
    ctx.CheckCall(
        [], {
            "input": 'sqlite:////workdir/java.db',
            "output": 'sqlite:////workdir/export.db',
        },
        volumes={tempdir: '/workdir'})

  # Check that corpus is exported.
  assert (tempdir / "java.db").is_file()
  assert (tempdir / "export.db").is_file()


if __name__ == '__main__':
  test.Main()
