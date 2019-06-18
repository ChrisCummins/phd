#!/usr/bin/env bash
#
# Build and update the docker image and push it to dockerhub.
set -eux

DOCKERHUB_USER="chriscummins"

publish_docker_image() {
  # The fully qualified path, WITHOUT the leading '//'.
  local package="$1"
  local target="$2"
  # The name of the image tag to push to dockerhub.
  local image_name="$3"

  # The fully qualified target to build.
  local tar_target="//$package:$target.tar"
  # The path of the tar archive built by bazel.
  local tar_path="bazel-bin/$package/$target.tar"

  echo "Building $tar_target image"
  ./tools/docker/phd_build/run.sh bazel build "$tar_target"
  ENTRYPOINT=docker ./tools/docker/phd_build/run.sh load -i "$tar_path"
  echo "Pushing docker image $image_name"

  # The name of docker image that is built by bazel.
  local load_image_name="bazel/$package:$target"
  # The name of the docker image to export.
  local export_image_name="$DOCKERHUB_USER/$image_name:latest"

  docker tag "$load_image_name" "$export_image_name"
  docker rmi "$load_image_name"
  docker push "$export_image_name"
}


main() {
  publish_docker_image \
      "experimental/deeplearning/deepsmith/java_fuzz" \
      "scrape_java_files_image" \
      "java_fuzz_scraper"

  publish_docker_image \
      "experimental/deeplearning/deepsmith/java_fuzz" \
      "mask_contentfiles_image" \
      "java_fuzz_mask_contentfiles"

  publish_docker_image \
      "experimental/deeplearning/deepsmith/java_fuzz" \
      "export_java_corpus_image" \
      "java_fuzz_export_corpus"
}
main $@
