"""Integration test for Java fuzz pipeline."""

import pathlib

from labm8 import dockerutil
from labm8 import test

FLAGS = test.FLAGS


def test_end_to_end_pipeline(tempdir: pathlib.Path):
  scrape_java_files_image = dockerutil.BazelPy3Image(
      'experimental/deeplearning/deepsmith/java_fuzz/scrape_java_files_image')
  split_contentfiles_image = dockerutil.BazelPy3Image(
      'experimental/deeplearning/deepsmith/java_fuzz/scrape_java_files_image')
  export_java_corpus_image = dockerutil.BazelPy3Image(
      'experimental/deeplearning/deepsmith/java_fuzz/scrape_java_files_image')

  # Step 1: Scrape a single repo from GitHub.
  with scrape_java_files_image.RunContext() as ctx:
    ctx.Run([], {
        "n": 1,
        "db": '////sqlite:////workdir/java.db',
    },
            volumes={
                tempdir: '/workdir',
                '/var/phd': '/var/phd',
            })

  # Check that contentfiles database is created.
  assert (tempdir / "java.db").is_file()

  # Step 2: Export a subset of the contentfiles database.
  with split_contentfiles_image.RunContext() as ctx:
    ctx.Run(
        [], {
            "max_repo_count": 1,
            "input": '////sqlite:////workdir/java.db',
            "output": '////sqlite:////workdir/subset.db',
        },
        volumes={tempdir: '/workdir'})

  # Check that contentfiles database is created.
  assert (tempdir / "java.db").is_file()
  assert (tempdir / "subset.db").is_file()

  # Step 3: Export Java corpus.
  with export_java_corpus_image.RunContext() as ctx:
    ctx.Run(
        [], {
            "n":
            1,
            "db":
            '////sqlite:////workdir/java.db',
            "outdir":
            "/workdir/corpus",
            "preprocessors":
            "datasets.github.scrape_repos.preprocessors.extractors:JavaStaticMethods",
        },
        volumes={tempdir: '/workdir'})

  # Check that corpus is exported.
  assert (tempdir / "java.db").is_file()
  assert (tempdir / "corpus").is_dir()


if __name__ == '__main__':
  test.Main()
