#!/usr/bin/env bash
set -eux

source ~/phd/.env

if [[ "$(uname)" == "Darwin" ]]; then
  uname="darwin"
else
  uname="k8"
fi

cd "$PHD"

# Build and package the app
bazel build -c opt //experimental/deeplearning/deepsmith/opencl_fuzz:opencl_fuzz_image

rm -f "$PHD/experimental/deeplearning/deepsmith/opencl_fuzz/opencl_fuzz_image-layer.tar"
cp "$PHD/bazel-bin/experimental/deeplearning/deepsmith/opencl_fuzz/opencl_fuzz_image-layer.tar" \
  "$PHD/experimental/deeplearning/deepsmith/opencl_fuzz"

cd "$PHD/experimental/deeplearning/deepsmith/opencl_fuzz"
# Note that the --squash argument requires experimental features. See:
# https://github.com/docker/docker-ce/blob/master/components/cli/experimental/README.md
docker build . -t opencl_fuzz --squash

# Export a compressed tarball of the image.
# docker save opencl_fuzz | gzip -c > \
#    "$PHD/experimental/deeplearning/deepsmith/opencl_fuzz/docker_image.tar.gz"
