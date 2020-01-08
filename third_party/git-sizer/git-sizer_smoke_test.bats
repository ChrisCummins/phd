#!/usr/bin/env bats
#
# Test running git-sizer.
#
source labm8/sh/test.sh

BIN="$(DataPath phd/third_party/git-sizer/git-sizer)"

@test "run version" {
  "$BIN" --version
}
