#!/usr/bin/env bash
#
# Build an optimized binary and copy it, along with all of its runfiles, to the
# target prefix.

set -eu

PREFIX=$HOME/.local

mkdir -pv "$PREFIX/bin"

bazel build -c opt //tools/format

rm -rvf "$PREFIX/bin/format" "$PREFIX/bin/format.runfiles"
cp -v bazel-bin/tools/format/format "$PREFIX/bin/format"
cp -vrL bazel-bin/tools/format/format.runfiles "$PREFIX/bin/format.runfiles"
