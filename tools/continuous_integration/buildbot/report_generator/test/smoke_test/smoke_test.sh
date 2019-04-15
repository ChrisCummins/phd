#!/usr/bin/env bash
#
# Test that report generator doesn't blow up.
#
set -eux
workdir="$(mktemp -d)"

# Tidy up.
cleanup() {
  rm -rvf "$workdir"
}
trap cleanup EXIT


# Main entry point.
main() {
  mkdir "$workdir"/repo
  git -C "$workdir"/repo init
  git -C "$workdir"/repo config user.email "example@name.com"
  git -C "$workdir"/repo config user.name "Example Name"
  git -C "$workdir"/repo commit --allow-empty -m "Hello, world"

  tools/continuous_integration/buildbot/report_generator/report_generator \
      --db="sqlite:///$workdir/db" \
      --host=testbed \
      --testlogs=tools/continuous_integration/buildbot/report_generator/test/data/testlogs \
      --repo="$workdir/repo" \
      || true

  # Run again to produce a second set of inputs.
  tools/continuous_integration/buildbot/report_generator/report_generator \
      --db="sqlite:///$workdir/db" \
      --host=testbed \
      --testlogs=tools/continuous_integration/buildbot/report_generator/test/data/testlogs \
      --repo="$workdir/repo"
}
main $@
