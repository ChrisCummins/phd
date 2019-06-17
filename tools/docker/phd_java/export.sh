#!/usr/bin/env bash
#
# Build and publish updated docker image.
set -eux

docker build -t phd_base_java $PHD/tools/docker/phd_base_java
docker tag phd_base_java chriscummins/phd_base_java:latest
docker push chriscummins/phd_base_java:latest
