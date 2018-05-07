#!/usr/bin/env bash

# doc_pdfs.sh - Build PDFs of documents and track them.
#
# Usage:
#
#     ./doc_pdfs.sh
#
set -eu

# Directory of this script.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run from the workspace root directory.
cd "$DIR/.."

bazel build //docs:all

for name in $(bazel query //docs:all | sed -r 's,^//docs:(.*)$,\1,'); do
  cp bazel-genfiles/docs/$name.pdf docs/$name.pdf
  chmod 664 docs/$name.pdf
  git add docs/$name.pdf
done
