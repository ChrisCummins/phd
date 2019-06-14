#!/usr/bin/env bash
#
# Test that Java file scraper doesn't explode.
#
set -eux
workdir="$(mktemp -d)"

# Tidy up.
cleanup() {
  rm -rvf "$workdir"
}
trap cleanup EXIT

# Scrape a single repository.
experimental/deeplearning/deepsmith/java_fuzz/scrape_java_files \
    --n=1 --db="sqlite:///$workdir/java.db"

# Check that database has been created.
test -f "$workdir/java.db"
