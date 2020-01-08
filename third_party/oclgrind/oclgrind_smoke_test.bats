#!/usr/bin/env bats
#
# Test running oclgrind.
#
source labm8/sh/test.sh

OCLGRIND="$(DataPath phd/third_party/oclgrind/oclgrind)"

@test "run help" {
  run "$OCLGRIND" --help
  [ "$status" -eq 0 ]
}

@test "run help from another directory" {
  cd "$TEST_TMPDIR"
  run "$OCLGRIND" --help
  [ "$status" -eq 0 ]
}
