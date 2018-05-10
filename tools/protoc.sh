#!/usr/bin/env bash

# protoc.sh - Run protoc on all proto directories to generate python code.
#
# Usage:
#
#     ./protoc.sh
#
set -eu

# The root of the repository.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$ROOT"
for file in $(git ls-files | grep '\.proto'); do
  dir="$(dirname $file)"
  protoc -I="$dir" -I="$ROOT" --python_out="$dir" "$file"
done
