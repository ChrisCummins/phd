#!/usr/bin/env bash

#
# buildifier.sh - Run buildifier on BUILD files. This modifiers BUILD files
# in-place.
#
# Usage:
#
#     ./buildifier
#
set -eu

# Directory of this script.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run from the workspace root directory.
cd "$DIR/.."

buildifier WORKSPACE
git ls-files | grep BUILD | xargs buildifier
