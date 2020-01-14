#!/usr/bin/env bats

source labm8/sh/test.sh

BIN=$(DataPath phd/third_party/llvm/clang)

@test "help" {
  run "$BIN" --help
}

@test "create LLVM module" {
  # Read a C program from stdin and print LLVM module to stdout.
  cat <<EOF | "$BIN" -xc - -o - -emit-llvm -S
int main() {
  return 5;
}
EOF
}
