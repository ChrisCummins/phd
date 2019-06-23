#!/usr/bin/env bash
set -eux

# Run the toy compiler to generate assembly.
../*/toy $1
base="${1%.*}"
test -f "$base".s
# Use GCC to generate the binary.
gcc "$base".s -o "$base"
test -f "$base"
