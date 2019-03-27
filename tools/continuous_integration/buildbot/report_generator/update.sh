#!/usr/bin/env bash
#
# Build and update docker iamge.
set -eux

main() {
  bazel build -c opt //tools/continuous_integration/buildbot/report_generator:image.tar
  docker load -i bazel-bin/tools/continuous_integration/buildbot/report_generator/image.tar
  docker tag bazel/tools/continuous_integration/buildbot/report_generator:image chriscummins/buildbot_testlogs_import
  docker push chriscummins/buildbot_testlogs_import
}
main $@
