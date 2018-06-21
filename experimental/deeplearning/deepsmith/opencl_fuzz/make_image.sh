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
mkdir -pv $PHD/experimental/deeplearning/deepsmith/opencl_fuzz/cache/model
mkdir -pv $PHD/experimental/deeplearning/deepsmith/opencl_fuzz/cache/corpus/encoded
mkdir -pv $PHD/experimental/deeplearning/deepsmith/opencl_fuzz/cache/corpus/preprocessed

MODEL_SRC="/mnt/cc/data/experimental/deeplearning/polyglot/clgen/model/71aef8088a45cd5e63c800bc025187a090c77103"
ENCODED_SRC="$(readlink $MODEL_SRC/corpus)"
PREPROCESSED_SRC="$(readlink $ENCODED_SRC/preprocessed)"

rsync -avh --delete --no-links "$ENCODED_SRC" \
    $PHD/experimental/deeplearning/deepsmith/opencl_fuzz/cache/corpus/encoded
rsync -avh --delete --no-links "$PREPROCESSED_SRC" \
    $PHD/experimental/deeplearning/deepsmith/opencl_fuzz/cache/corpus/preprocessed
rsync -avh --delete --no-links "$MODEL_SRC" --exclude samples \
    $PHD/experimental/deeplearning/deepsmith/opencl_fuzz/cache/model

# The name of the package, *without* leading slashes.
package="experimental/deeplearning/deepsmith/opencl_fuzz"
target="opencl_fuzz_image"

# Build and package the app
bazel build -c opt "//$package":"$target"

rm -f "$PHD/experimental/deeplearning/deepsmith/opencl_fuzz/$target-layer.tar"
cp "$PHD/bazel-bin/$package/$target-layer.tar" \
    $PHD/experimental/deeplearning/deepsmith/opencl_fuzz

cd $PHD/experimental/deeplearning/deepsmith/opencl_fuzz
# Note that the --squash argument requires experimental features. See:
# https://github.com/docker/docker-ce/blob/master/components/cli/experimental/README.md
sudo docker build . -t opencl_fuzz --squash

# Export a compressed tarball of the image.
docker save opencl_fuzz | gzip -c > \
    $PHD/experimental/deeplearning/deepsmith/opencl_fuzz/docker_image.tar.gz
