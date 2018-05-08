#!/usr/bin/env bash

# cleanup.sh - Remove files generated during archive install.
#
# This removes the files created by ./install.sh. Note that packages that are
# installed through a package manager (for example, using apt-get) are not
# removed, since we do not know if they were installed by our script, or by
# the user. This script may be run repeatedly.
#
# Usage:
#
#     ./cleanup.sh
#
set -eux

# Directory of this script.
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

main() {
  # Run from the root of the package.
  cd "$DIR"

  paths_to_delete=(
    "$DIR/build"
    ~/.cache/clgen/models/TODO
  )

  for path in ${paths_to_delete[*]}; do
    echo "$path"
    # rm -rfv "$path"
  done
}

main $@
