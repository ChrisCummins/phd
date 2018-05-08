#!/usr/bin/env bash

# run.sh - Run the experiments.
#
# Usage:
#
#     ./run.sh
#
set -eux


main() {
  # Run from the artifact_evaluation root directory.
  cd "$(dirname "${BASH_SOURCE[0]}")/.."

  # Disable unbound variable errors, since ./build/phd/.env checks whether
  # $VIRTUAL_ENV is set.
  set +u
  source "$DIR/build/phd/.env"
  # Re-enable unbound variable errors.
  set -u

  clgen sample ~/data/models/github-512x2x50-greedy.json
}
main $@
