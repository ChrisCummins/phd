#!/usr/bin/env bash
#
# Test that `export_source_tree` can export itself, and the result can be built.
#
set -eux

TMPDIR="$(mktemp -d)"

# Tidy up.
cleanup() {
  rm -rf "$TMPDIR"
}
trap cleanup EXIT

tools/source_tree/export_source_tree \
    --targets=//tools/source_tree:export_source_tree \
    --destination="$TMPDIR"

find "$TMPDIR" -type f

test -f "$TMPDIR/tools/source_tree/export_source_tree.py"

cd "$TMPDIR"
# Build the exported source tree.
./configure --noninteractive
test -f bootstrap.sh
./bazel_wrapper.py build //tools/source_tree:export_source_tree

# Tidy up.
./bazel_wrapper.py clean --expunge
