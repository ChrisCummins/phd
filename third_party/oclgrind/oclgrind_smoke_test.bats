#!/usr/bin/env bats
#
# Test running oclgrind.
#
source labm8/sh/test.sh

OCLGRIND="$(DataPath phd/third_party/oclgrind/oclgrind)"

@test "run help" {
  "$OCLGRIND" --help
}

@test "run help from another directory" {
  cd "$TEST_TMPDIR"
  "$OCLGRIND" --help
}

@test "run hello binary" {
  "$OCLGRIND" --check-api --data-races "$(DataPath phd/third_party/opencl/examples/hello)"
}
