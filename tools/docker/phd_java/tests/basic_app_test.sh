set -eux

main() {
  local tar_path="tools/docker/phd_java/tests/basic_app_test_image.tar"
  local image_name="bazel/tools/docker/phd_java/tests:basic_app_test_image"

  test -f "$tar_path"

  docker load -i "$tar_path"
  docker run "$image_name"
  docker rmi --force "$image_name"
}
main $@
