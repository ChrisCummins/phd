#!/usr/bin/env bats

source labm8/sh/test.sh

BIN=$(DataPath phd/third_party/llvm/opt)

@test "help" {
  run "$BIN" --help
}