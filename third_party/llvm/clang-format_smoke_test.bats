#!/usr/bin/env bats

source labm8/sh/test.sh

BIN=$(DataPath phd/third_party/llvm/clang-format)

@test "help" {
  $BIN --help
}

@test "read stdin" {
  echo "hello" | $BIN
  false
}
