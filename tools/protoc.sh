#!/usr/bin/env bash

# protoc.sh - Run protoc on all proto directories to generate python code.
#
# Usage:
#
#     ./protoc.sh
#
set -eu

# Directory of this script.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run from the workspace root directory.
cd "$DIR/.."

for dir in $(find . -name proto -type d); do
  if compgen -G "$dir/*.proto" > /dev/null; then
    protoc -I$dir --python_out=$dir $dir/*.proto
  fi
done
