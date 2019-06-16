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
  # Scrape a single repository.
  experimental/deeplearning/deepsmith/java_fuzz/scrape_java_files \
      --n=1 --db="sqlite:///$workdir/java.db"

  # Check that database has been created.
  test -f "$workdir/java.db"

  # Export a subset of the content files.
  experimental/deeplearning/deepsmith/java_fuzz/split_contentfiles \
      --n=1 --input="sqlite:///$workdir/java.db" --output="sqlite:///$workdir/subset.db"

  # Check that a new database has been created.
  test -f "$workdir/java.db"
  test -f "$workdir/subset.db"

  # Pre-process.
  experimental/deeplearning/deepsmith/java_fuzz/export_java_corpus \
      --db="sqlite:///$workdir/java.db" --outdir="$workdir"/corpus \
      --preprocessors="datasets.github.scrape_repos.preprocessors.extractors:JavaStaticMethods"

  test -d "$workdir/corpus"
}
main $@
