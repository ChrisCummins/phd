#!/usr/bin/env bash
#
# Shared bash functions for testing.
#
# Usage:  source labm8/sh/test.sh

# Make a temporary directory, with an optional suffix.
# The calling code is responsible for removing this directory when done.
MakeTemporaryDirectory() {
  local suffix="$1"
  local workdir="$(mktemp -d --suffix=_labm8_sh$suffix)"
  echo "$workdir"
}
