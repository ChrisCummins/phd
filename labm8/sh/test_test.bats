#!/usr/bin/env bats

source labm8/sh/test.sh

tempdir="$(MakeTemporaryDirectory)"

setup() {
  mkdir -p "$tempdir"
}

teardown() {
  rm -rfv "$tempdir"
}

@test "tempdir exists" {
  test -d "$tempdir"
}

@test "tempdir is writable" {
  touch "$tempdir/a"
  test -f "$tempdir/a"
}
