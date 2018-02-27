#!/usr/bin/env bash
#
# buildifier.sh - Run buildifier on BUILD files.
#
# Usage:
#
#     ./buildifier
#
set -eux

# Directory of this script.
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run from the workspace root directory.
cd "$DIR/.."

buildifier WORKSPACE
git ls-files | grep BUILD | xargs buildifier
