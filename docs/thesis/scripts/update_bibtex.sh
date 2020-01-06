#!/usr/bin/env bash
set -eux

BIBTEX_FILE_TO_IMPORT="$HOME/Dropbox/Apps/Mendeley/library.bib"
PHD_REPO="$HOME/phd"

main() {
  test -f bibliography.bib
  test -f "$BIBTEX_FILE_TO_IMPORT"
  test -f "$PHD_REPO/WORKSPACE"
  cp "$BIBTEX_FILE_TO_IMPORT" bibliography.bib
  cwd="$(pwd)"
  cd "$PHD_REPO" && bazel run //docs:minimize_bibtex -- --bibtex_path="$cwd/bibliography.bib"
  cd "$PHD_REPO" && bazel run //docs:deacronym_bibtex -- --bibtex_path="$cwd/bibliography.bib"
}
main $@
