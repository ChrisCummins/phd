#!/usr/bin/env bash

# run.sh - Run the experiments.
#
# Usage:
#
#     ./run.sh
#
set -eu


# The artficat_evaluation root directory.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"


main() {
  # Run from the artifact_evaluation root directory.
  cd "$ROOT"

  # Disable unbound variable errors, since ./build/phd/.env checks whether
  # $VIRTUAL_ENV is set.
  set +u
  # TODO: source "$ROOT/build/phd/.env"
  # Re-enable unbound variable errors.
  set -u

  python "$ROOT/02_evaluate_harness/experiments.py"
}
main $@
