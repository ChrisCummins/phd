#!/usr/bin/env bash
#
# Build and publish the chriscummins/llvm2graph docker image.
#
set -eux

VERSION=$(cat version.txt)

# Build the docker archive.
bazel build deeplearning/ml4pl/graphs/unlabelled/llvm2graph:llvm2graph_image.tar --host_force_python=PY2

# Load the docker archive.
docker load <bazel-bin/deeplearning/ml4pl/graphs/unlabelled/llvm2graph/llvm2graph_image.tar

# Sanity check.
docker run bazel/deeplearning/ml4pl/graphs/unlabelled/llvm2graph:llvm2graph_image

# Tag and upload the docker image.
docker tag bazel/deeplearning/ml4pl/graphs/unlabelled/llvm2graph:llvm2graph_image chriscummins/llvm2graph:${VERSION}
docker push chriscummins/llvm2graph:${VERSION}

# Create the :latest version tag.
docker tag chriscummins/llvm2graph:${VERSION} chriscummins/llvm2graph:latest
docker push chriscummins/llvm2graph:latest

# Tidy up.
docker image rmi bazel/deeplearning/ml4pl/graphs/unlabelled/llvm2graph:llvm2graph_image
