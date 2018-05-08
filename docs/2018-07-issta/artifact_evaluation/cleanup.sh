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


main() {
    # Run from the artifact_evaluation root directory.
  cd "$(dirname "${BASH_SOURCE[0]}")"

  paths_to_delete=(
    "./build"
    # ~/.cache/clgen/models/TODO
  )

  for path in ${paths_to_delete[*]}; do
    rm -rf "$path"
  done
}

main $@
