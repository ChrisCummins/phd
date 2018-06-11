#!/usr/bin/env bash
# Make a standalone binary release of CLgen.
# This uses whatever the configuration of the host repository/machine. E.g.
# a mac/linux build, with/without CUDA support.
set -eux

source ~/phd/.env

# Requirements: python3.6
# TODO: homebrew llvm, until config.pbtxt is dropped.

if [[ "$(uname)" == "Darwin" ]]; then
  uname="darwin"
else
  uname="k8"
fi

date="$(date +%s)"

tarball="clgen_""$uname""_$date"

bazel build //deeplearning/clgen:clgen_test
cd bazel-phd/bazel-out/"$uname"-py3-opt/bin/deeplearning/clgen
tar cjvfh "$tarball".tar.bz2 \
  --exclude '*.runfiles_manifest' \
  --exclude '*.intellij-info.txt' \
  --exclude 'MANIFEST' \
  --exclude '__pycache__' \
  clgen_test clgen_test.runfiles
mv "$tarball".tar.bz2 $PHD
