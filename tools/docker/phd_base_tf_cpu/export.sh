#!/usr/bin/env bash
#
# Build and publish updated docker image.
set -eux

version=$(cat version.txt)

docker build -t phd_base_tf_cpu "$PHD"/tools/docker/phd_base_tf_cpu
docker tag phd_base_tf_cpu chriscummins/phd_base_tf_cpu:latest
docker tag phd_base_tf_cpu chriscummins/phd_base_tf_cpu:"$version"
docker push chriscummins/phd_base_tf_cpu:latest
docker push chriscummins/phd_base_tf_cpu:"$version"
docker rmi phd_base_tf_cpu
