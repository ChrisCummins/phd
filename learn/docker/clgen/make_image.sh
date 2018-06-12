#!/usr/bin/env bash
# Make a standalone binary release of CLgen.
# This uses whatever the configuration of the host repository/machine. E.g.
# a mac/linux build, with/without CUDA support.
set -eux

source ~/phd/.env

if [[ "$(uname)" == "Darwin" ]]; then
  uname="darwin"
else
  uname="k8"
fi

cd "$PHD"
package="deeplearning/clgen/preprocessors"
target="cxx_test"

bazel build -c opt "$package":"$target"

cd bazel-phd/bazel-out/"$uname"-py3-opt/bin/"$package"
tar cjvfh clgen.tar.bz2 \
  --exclude '*.runfiles_manifest' \
  --exclude '*.intellij-info.txt' \
  --exclude 'MANIFEST' \
  --exclude '__pycache__' \
  "$target" "$target".runfiles
mv clgen.tar.bz2 $PHD/learn/docker/clgen

cd $PHD/learn/docker/clgen

sudo docker build -t clgen .

sudo docker run -it clgen /bin/bash
