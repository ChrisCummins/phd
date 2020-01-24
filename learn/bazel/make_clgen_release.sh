#!/usr/bin/env bash
# Make a standalone binary release of CLgen.
# This uses whatever the configuration of the host repository/machine. E.g.
# a mac/linux build, with/without CUDA support.
set -eux

source ~/phd/.env

# Requirements: python3.6

if [[ "$(uname)" == "Darwin" ]]; then
  uname="darwin"
else
  uname="k8"
fi

date="$(date +%s)"

target="clgen"

bazel build //deeplearning/clgen:"$target"
cd bazel-phd/bazel-out/"$uname"-py3-opt/bin/deeplearning/clgen
tar cjvfh "$target".tar.bz2 \
  --exclude '*.runfiles_manifest' \
  --exclude '*.intellij-info.txt' \
  --exclude 'MANIFEST' \
  --exclude '__pycache__' \
  "$target" "$target".runfiles
mv "$target".tar.bz2 $PHD
