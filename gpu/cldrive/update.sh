#!/usr/bin/env bash
#
# Script to build and update the docker hub image.
set -eux

main() {
  bazel build -c opt //gpu/cldrive:cldrive_image.tar
  docker load -i bazel-bin/gpu/cldrive/cldrive_image.tar

  # Test that it works.
  docker run \
      -v/etc/OpenCL/vendors:/etc/OpenCL/vendors \
      -v/opt/intel/:/opt/intel/ \
      -v/etc/OpenCL/vendors/intel64.icd:/etc/OpenCL/vendors/intel64.icd \
      -v$PWD:/cwd \
      bazel/gpu/cldrive:cldrive_image \
      --clinfo

  docker tag bazel/gpu/cldrive:cldrive_image chriscummins/cldrive:latest
  docker push chriscummins/cldrive:latest
}
main $@
