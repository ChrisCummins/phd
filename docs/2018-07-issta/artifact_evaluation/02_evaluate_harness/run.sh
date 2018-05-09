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

  # Create a working copy of the pre-populated datastore.
  mkdir -pv "$ROOT/02_evaluate_harness/run"
  cp "$ROOT/02_evaluate_harness/data/datastore.db" \
      "$ROOT/02_evaluate_harness/output/datastore.db"

  python "$ROOT/02_evaluate_harness/experiments.py"
}
main $@
