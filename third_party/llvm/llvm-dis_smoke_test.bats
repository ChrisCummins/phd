#!/usr/bin/env bats

source labm8/sh/test.sh

BIN=$(DataPath phd/third_party/llvm/llvm-dis)

@test "help" {
  run "$BIN" --help
}
