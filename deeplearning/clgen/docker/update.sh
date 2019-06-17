#!/usr/bin/env bash
#
# Build and update the CLgen docker image and push it to dockerhub.
set -eux

main() {
  # clgen
  set +e
  ./tools/docker/phd_build/run.sh bazel run //deeplearning/clgen/docker:clgen
  set -e
  docker tag bazel/deeplearning/clgen/docker:clgen chriscummins/clgen:latest
  docker push chriscummins/clgen:latest

  # clgen_preprocess
  ./tools/docker/phd_build/run.sh bazel build //deeplearning/clgen/docker:clgen_preprocess.tar
  ENTRYPOINT=docker ./tools/docker/phd_build/run.sh load -i \
      bazel-bin/deeplearning/clgen/docker/clgen_preprocess.tar
  docker tag bazel/deeplearning/clgen/docker:clgen_preprocess chriscummins/clgen_preprocess
  docker rmi bazel/deeplearning/clgen/docker:clgen_preprocess
  docker push chriscummins/clgen_preprocess
}
main $@
