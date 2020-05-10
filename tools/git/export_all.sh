#!/usr/bin/env bash
#
# Run all exports_repo() targets to update all exported subprojects.
#
# Usage:
#
#     $ ./tools/git/export_all.sh
set -eu

for target in $(bazel query 'kind(exports_repo, //...)'); do
  echo $target
  bazel run $target --define=workspace=$(pwd)
done
