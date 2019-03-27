#!/usr/bin/env bash
#
# Build and update the CLgen docker image and push it to dockerhub.
set -eux

main() {
  set +e
  ./tools/docker/phd_build/run.sh bazel run //deeplearning/clgen/docker:clgen
  set -e
  docker tag bazel/deeplearning/clgen/docker:clgen chriscummins/clgen:latest
  docker push chriscummins/clgen:latest
}
main $@
