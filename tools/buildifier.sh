#!/usr/bin/env bash

# buildifier.sh - Run buildifier on BUILD files. This modifiers BUILD files
# in-place.
#
# Usage:
#
#     ./buildifier.sh
#
set -eu

# Directory of this script.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run from the workspace root directory.
cd "$DIR/.."

buildifier WORKSPACE
# Git ls-files will return files that do not exist if a file has been removed
# but not committed.
for f in $(git ls-files | grep BUILD); do
  if [[ -f "$f" ]]; then
    buildifier "$f"
  fi
done
