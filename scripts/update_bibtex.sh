#!/usr/bin/env bash
set -eux

BIBTEX_FILE_TO_IMPORT="$HOME/docs/library.bib"
PHD_REPO="$HOME/phd"

main() {
  test -f refs.bib
  test -f "$BIBTEX_FILE_TO_IMPORT"
  test -f "$PHD_REPO/WORKSPACE"
  # cp "$BIBTEX_FILE_TO_IMPORT" refs.bib
  cwd="$(pwd)"
  cd "$PHD_REPO" && bazel run //docs:minimize_bibtex -- --bibtex_path="$cwd/refs.bib"
}
main $@
