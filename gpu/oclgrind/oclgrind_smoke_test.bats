#!/usr/bin/env bats
#
# Test running cldrive under oclgrind.
#
source labm8/sh/test.sh

OCLGRIND="$(DataPath phd/third_party/oclgrind/oclgrind)"

@test "run clinfo" {
  run "$OCLGRIND" "$(DataPath phd/gpu/clinfo/clinfo)"
  [ "$status" -eq 0 ]
}

@test "run oclgrind_working" {
  run "$OCLGRIND" "$(DataPath phd/gpu/oclgrind/test/data/oclgrind_working)"
  [ "$status" -eq 0 ]
}
