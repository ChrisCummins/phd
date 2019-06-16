#!/usr/bin/env bash
#
# Build and update the docker image and push it to dockerhub.
set -eu

# Fully qualified path, WITHOUT the '//' prefix
BAZEL_TARGET="experimental/deeplearning/deepsmith/java_fuzz:scrape_java_files_image"
IMAGE_NAME="java_fuzz_scraper"
DOCKERHUB_USER="chriscummins"

publish_docker_image() {
  local bazel_target="$1"
  local image_name="$2"

  echo "Building bazel target $bazel_target"
  set +e
  ./tools/docker/phd_build/run.sh bazel run "$bazel_target"
  set -e
  echo "Pushing docker image $image_name"
  docker tag bazel/"$bazel_target" "$DOCKERHUB_USER"/"$image_name":latest
  docker push "$DOCKERHUB_USER"/"$image_name":latest
}


main() {
  publish_docker_image \
      "experimental/deeplearning/deepsmith/java_fuzz:scrape_java_files_image" \
      "java_fuzz_scraper"

  publish_docker_image \
      "experimental/deeplearning/deepsmith/java_fuzz:split_contentfiles" \
      "java_fuzz_split_contentfiles"

  publish_docker_image \
      "experimental/deeplearning/deepsmith/java_fuzz:export_java_corpus" \
      "java_fuzz_export_corpus"
}
main $@
