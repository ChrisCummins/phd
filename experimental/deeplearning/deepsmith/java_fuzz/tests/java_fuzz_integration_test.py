"""Integration test for Java fuzz pipeline."""
import pathlib

from datasets.github.testing.access_token import ACCESS_TOKEN
from datasets.github.testing.requires_access_token import requires_access_token
from labm8.py import dockerutil
from labm8.py import test

FLAGS = test.FLAGS


@test.XFail(reason="Failure during pre-processingq. Fix me.")
@requires_access_token
def test_end_to_end_pipeline(tempdir: pathlib.Path):
  scrape_java_files_image = dockerutil.BazelPy3Image(
    "experimental/deeplearning/deepsmith/java_fuzz/scrape_java_files_image"
  )
  mask_contentfiles_image = dockerutil.BazelPy3Image(
    "experimental/deeplearning/deepsmith/java_fuzz/mask_contentfiles_image"
  )
  export_java_corpus_image = dockerutil.BazelPy3Image(
    "experimental/deeplearning/deepsmith/java_fuzz/export_java_corpus_image"
  )
  preprocess_java_corpus_image = dockerutil.BazelPy3Image(
    "experimental/deeplearning/deepsmith/java_fuzz/preprocess_java_corpus_image"
  )
  re_preprocess_java_methods_image = dockerutil.BazelPy3Image(
    "experimental/deeplearning/deepsmith/java_fuzz/re_preprocess_java_methods_image"
  )
  encode_java_corpus_image = dockerutil.BazelPy3Image(
    "experimental/deeplearning/deepsmith/java_fuzz/encode_java_corpus_image"
  )

  # Step 1: Scrape a single repo from GitHub.
  with scrape_java_files_image.RunContext() as ctx:
    ctx.CheckCall(
      [],
      {
        "n": 1,
        "db": "sqlite:////workdir/java.db",
        "github_access_token": ACCESS_TOKEN,
      },
      volumes={tempdir: "/workdir",},
      timeout=600,
    )

  # Check that contentfiles database is created.
  assert (tempdir / "java.db").is_file()

  # Step 2: Mask a subset of the contentfiles database.
  with mask_contentfiles_image.RunContext() as ctx:
    ctx.CheckCall(
      [],
      {"db": "sqlite:////workdir/java.db", "max_repo_count": 1,},
      volumes={tempdir: "/workdir"},
      timeout=300,
    )

  # Check that contentfiles database is still there.
  assert (tempdir / "java.db").is_file()

  # Step 3: Export Java corpus.
  with export_java_corpus_image.RunContext() as ctx:
    ctx.CheckCall(
      [],
      {
        "input": "sqlite:////workdir/java.db",
        "output": "sqlite:////workdir/export.db",
      },
      volumes={tempdir: "/workdir"},
      timeout=600,
    )

  # Check that corpus is exported.
  assert (tempdir / "java.db").is_file()
  assert (tempdir / "export.db").is_file()

  # Step 4: Preprocess Java corpus.
  with preprocess_java_corpus_image.RunContext() as ctx:
    ctx.CheckCall(
      [],
      {
        "input": "sqlite:////workdir/export.db",
        "output": "sqlite:////workdir/preprocessed.db",
      },
      volumes={tempdir: "/workdir"},
      timeout=600,
    )

  # Check that corpus is exported.
  assert (tempdir / "export.db").is_file()
  assert (tempdir / "preprocessed.db").is_file()

  # Step 5: Re-Preprocess Java methods.
  with re_preprocess_java_methods_image.RunContext() as ctx:
    ctx.CheckCall(
      [],
      {
        "input": "sqlite:////workdir/exported.db",
        "input_pp": "sqlite:////workdir/preprocessed.db",
        "outdir": "/workdir/re_preprocessed",
      },
      volumes={tempdir: "/workdir"},
      timeout=600,
    )

  # Check that corpus is exported.
  assert (tempdir / "preprocessed.db").is_file()
  assert (tempdir / "re_preprocessed").is_dir()

  # Step 6: Encode Java methods.
  with encode_java_corpus_image.RunContext() as ctx:
    ctx.CheckCall(
      [],
      {
        "input": "sqlite:////workdir/preprocessed.db",
        "output": "sqlite:////workdir/encoded.db",
      },
      volumes={tempdir: "/workdir"},
      timeout=600,
    )

  # Check that corpus is encoded.
  assert (tempdir / "preprocessed.db").is_file()
  assert (tempdir / "encoded.db").is_file()


if __name__ == "__main__":
  test.Main()
