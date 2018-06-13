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

# Export pre-trained CLgen cache.
mkdir -pv $PHD/experimental/deeplearning/clgen/docker_worker/cache/model
mkdir -pv $PHD/experimental/deeplearning/clgen/docker_worker/cache/corpus/encoded
mkdir -pv $PHD/experimental/deeplearning/clgen/docker_worker/cache/corpus/preprocessed

MODEL_SRC="/mnt/cc/data/experimental/deeplearning/polyglot/clgen/model/cb9d1984a341036733c91d96c07a5bcff276b829"
ENCODED_SRC="/mnt/cc/data/experimental/deeplearning/polyglot/clgen/corpus/encoded/0e55c550b119bcfca0e9c8e09e69466fcc48283a"
PREPROCESSED_SRC="/mnt/cc/data/experimental/deeplearning/polyglot/clgen/corpus/preprocessed/6286c1875f8bb046b95689bb4935e9bd42301925"

rsync -avh --delete --no-links "$ENCODED_SRC" \
    $PHD/experimental/deeplearning/clgen/docker_worker/cache/corpus/encoded
rsync -avh --delete --no-links "$PREPROCESSED_SRC" \
    $PHD/experimental/deeplearning/clgen/docker_worker/cache/corpus/preprocessed
rsync -avh --delete --no-links "$MODEL_SRC" --exclude samples \
    $PHD/experimental/deeplearning/clgen/docker_worker/cache/model

# TODO(cec): Once support for Contentfiles ID is implemented, remove this.
CONTENTFILES_SRC="/mnt/cc/data/datasets/github/corpuses/java/"
rsync -avh --delete --no-links "$CONTENTFILES_SRC" \
    $PHD/experimental/deeplearning/clgen/docker_worker/corpus

# Build and package CLgen.
package="deeplearning/clgen"
target="clgen"
bazel build -c opt "$package":"$target"
cd bazel-phd/bazel-out/"$uname"-py3-opt/bin/"$package"
tar cjvfh clgen.tar.bz2 \
  --exclude '*.runfiles_manifest' \
  --exclude '*.intellij-info.txt' \
  --exclude 'MANIFEST' \
  --exclude '__pycache__' \
  "$target" "$target".runfiles
mv clgen.tar.bz2 $PHD/experimental/deeplearning/clgen/docker_worker

cd $PHD/experimental/deeplearning/clgen/docker_worker

sudo docker build -t clgen_pretrained .
sudo docker run -it clgen_pretrained /bin/bash
