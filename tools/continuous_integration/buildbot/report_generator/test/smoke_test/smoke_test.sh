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
  tools/continuous_integration/buildbot/report_generator/report_generator \
      --db="sqlite:///$workdir/db" \
      --host=testbed \
      --testlogs=tools/continuous_integration/buildbot/report_generator/test/data/testlogs \
      || true

  # Run again to produce a second set of inputs.
  tools/continuous_integration/buildbot/report_generator/report_generator \
      --db="sqlite:///$workdir/db" \
      --host=testbed \
      --testlogs=tools/continuous_integration/buildbot/report_generator/test/data/testlogs
}
main $@
