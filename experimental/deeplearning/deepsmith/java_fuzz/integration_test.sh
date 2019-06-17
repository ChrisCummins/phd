#!/usr/bin/env bash
#
# Test a sequence of commands.
#
set -eux
workdir="$(mktemp -d)"

# Tidy up.
cleanup() {
  rm -rvf "$workdir"
}
trap cleanup EXIT

main() {
  docker load -i experimental/deeplearning/deepsmith/java_fuzz/scrape_java_files_image.tar
  docker load -i experimental/deeplearning/deepsmith/java_fuzz/split_contentfiles_image.tar
  docker load -i experimental/deeplearning/deepsmith/java_fuzz/export_java_corpus_image.tar

  # Scrape a single repository.
  # We share /var/phd so that /var/phd/github_access_token.txt is available
  # to the scraper image.
  docker run -v"$workdir":/workdir -v/var/phd:/var/phd bazel/experimental/deeplearning/deepsmith/java_fuzz:scrape_java_files_image \
      --n=1 --db="sqlite:////workdir/java.db"

  # Check that database has been created.
  test -f "$workdir/java.db"

  # Export a subset of the content files.
  docker run -v"$workdir":/workdir bazel/experimental/deeplearning/deepsmith/java_fuzz:split_contentfiles_image \
      --n=1 --input="sqlite:////workdir/java.db" --output="sqlite:////workdir/subset.db"

  # Check that a new database has been created.
  test -f "$workdir/java.db"
  test -f "$workdir/subset.db"

  # Pre-process.
  docker run -v"$workdir":/workdir \
      bazel/experimental/deeplearning/deepsmith/java_fuzz:export_java_corpus_image \
      --db="sqlite:////workdir/java.db" --outdir=/workdir/corpus \
      --preprocessors="datasets.github.scrape_repos.preprocessors.extractors:JavaStaticMethods"

  test -d "$workdir/corpus"

  # Remove the docker images.
  docker rmi bazel/experimental/deeplearning/deepsmith/java_fuzz:scrape_java_files_image
  docker rmi bazel/experimental/deeplearning/deepsmith/java_fuzz:split_contentfiles_image
  docker rmi bazel/experimental/deeplearning/deepsmith/java_fuzz:export_java_corpus_image
}
main $@
