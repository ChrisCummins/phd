#!/usr/bin/env bash
#
# Build and update the docker image and push it to dockerhub.
set -eux

# Fully qualified path, WITHOUT the '//' prefix
BAZEL_TARGET="experimental/deeplearning/deepsmith/java_fuzz:scrape_java_files_image"
IMAGE_NAME="java_fuzz_scraper"
DOCKERHUB_USER="chriscummins"

main() {
  set +e
  ./tools/docker/phd_build/run.sh bazel run "$BAZEL_TARGET"
  set -e
  docker tag bazel/"$BAZEL_TARGET" "$DOCKERHUB_USER"/"$IMAGE_NAME":latest
  docker push "$DOCKERHUB_USER"/"$IMAGE_NAME":latest
}
main $@
