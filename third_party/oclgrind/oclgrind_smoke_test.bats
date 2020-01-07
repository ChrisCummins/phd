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

